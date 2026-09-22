from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .ocr_engine import OCRBox, RapidOCRProvider
from .settings import Settings


PTS_RE = re.compile(r"pts_time:([-+]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class SubtitleBox:
    text: str
    score: float
    bbox: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return x + width / 2, y + height / 2


@dataclass
class FrameOCR:
    requested_ms: int
    pts_ms: int
    pts_guard_ok: bool
    text: str
    confidence: float
    bbox: tuple[float, float, float, float] | None
    boxes: list[SubtitleBox]
    sharpness: float
    ocr_ms: float


@dataclass(frozen=True)
class DecodedFrame:
    requested_ms: int
    pts_ms: int
    image: Image.Image


def probe_duration_ms(video_path: Path, timeout: int = 30) -> int:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("ffprobe 不可用或执行超时") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 无法读取视频时长: {result.stderr.strip()}")
    try:
        seconds = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe 未返回有效视频时长") from exc
    if seconds <= 0:
        raise RuntimeError("视频时长无效")
    return round(seconds * 1000)


def _range_with_end(start: int, end: int, interval: int) -> set[int]:
    points = set(range(start, end + 1, interval))
    points.add(start)
    points.add(end)
    return points


def cue_window(segment: dict[str, Any], settings: Settings) -> tuple[int, int]:
    start = int(segment["start_ms"]) + settings.trim_start_ms
    end = int(segment["end_ms"]) - settings.trim_end_ms
    if start > end:
        midpoint = round((int(segment["start_ms"]) + int(segment["end_ms"])) / 2)
        return midpoint, midpoint
    return start, end


def interior_anchors(segment: dict[str, Any], settings: Settings) -> list[int]:
    start, end = cue_window(segment, settings)
    if start == end:
        return [start]
    if settings.min_inner_candidates <= 1:
        return [round((start + end) / 2)]
    return sorted({round(start + index * (end - start) / (settings.min_inner_candidates - 1))
                   for index in range(settings.min_inner_candidates)})


def candidate_points(duration_ms: int, segments: list[dict[str, Any]],
                     settings: Settings) -> list[int]:
    if duration_ms < 0:
        raise ValueError("视频时长不能为负数")
    points = _range_with_end(0, duration_ms, settings.grid_ms)
    for segment in segments:
        start = max(0, int(segment["start_ms"]) - settings.padding_ms)
        end = min(duration_ms, int(segment["end_ms"]) + settings.padding_ms)
        points.update(_range_with_end(start, end, settings.dense_ms))
        points.update(max(0, min(duration_ms, value))
                      for value in interior_anchors(segment, settings))
    return sorted(points)


def decode_frame(video_path: Path, requested_ms: int, timeout: int = 60) -> tuple[Image.Image, int]:
    seconds = f"{requested_ms / 1000:.3f}"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(video_path),
        "-frames:v", "1", "-vf", f"select=gte(t\\,{seconds}),showinfo",
        "-f", "image2pipe", "-vcodec", "png", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("FFmpeg 不可用或执行超时") from exc
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError("FFmpeg 无法解码请求帧")
    match = PTS_RE.search(result.stderr.decode("utf-8", errors="replace"))
    if not match:
        raise RuntimeError("FFmpeg 未返回真实帧 PTS")
    try:
        image = Image.open(BytesIO(result.stdout)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError("FFmpeg 输出不是有效图片") from exc
    return image, round(float(match.group(1)) * 1000)


def probe_frame_pts(video_path: Path, timeout: int = 900) -> list[tuple[int, int]]:
    """Return (decoded frame index, real PTS ms) without decoding image pixels."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("ffprobe 无法读取视频帧时间戳") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 无法读取视频帧时间戳: {result.stderr.strip()}")
    try:
        raw_frames = json.loads(result.stdout)["frames"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe 未返回视频帧时间戳") from exc
    frames = []
    for index, item in enumerate(raw_frames):
        raw_pts = item.get("best_effort_timestamp_time") if isinstance(item, dict) else None
        if raw_pts is None:
            continue
        try:
            pts_ms = round(float(raw_pts) * 1000)
        except (TypeError, ValueError):
            continue
        if pts_ms >= 0:
            frames.append((index, pts_ms))
    if not frames:
        raise RuntimeError("ffprobe 未返回有效视频帧时间戳")
    return frames


def _map_requests_to_frames(
    requested_points: Iterable[int], frame_pts: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Map requests to the first frame at or after each time and deduplicate PTS."""
    ordered = sorted(frame_pts, key=lambda item: (item[1], item[0]))
    times = [item[1] for item in ordered]
    by_pts: dict[int, tuple[int, int, int]] = {}
    for requested_ms in sorted(set(requested_points)):
        position = bisect_left(times, requested_ms)
        if position >= len(ordered):
            continue
        frame_index, pts_ms = ordered[position]
        current = by_pts.get(pts_ms)
        candidate = (frame_index, requested_ms, pts_ms)
        if current is None or abs(pts_ms - requested_ms) < abs(pts_ms - current[1]):
            by_pts[pts_ms] = candidate
    return sorted(by_pts.values(), key=lambda item: item[0])


def _balanced_select_expression(frame_indexes: list[int]) -> str:
    """Build a shallow expression tree; FFmpeg rejects long left-deep sums."""
    expressions = [f"eq(n\\,{index})" for index in frame_indexes]
    if not expressions:
        raise ValueError("目标帧不能为空")
    while len(expressions) > 1:
        combined = []
        for position in range(0, len(expressions), 2):
            if position + 1 == len(expressions):
                combined.append(expressions[position])
            else:
                combined.append(f"({expressions[position]})+({expressions[position + 1]})")
        expressions = combined
    return expressions[0]


def _cfr_targets(
    video_path: Path, requested_points: Iterable[int], timeout: int = 30,
) -> list[tuple[int, int, int]] | None:
    """Map timestamps from CFR metadata without scanning every decoded frame."""
    if not video_path.is_file():
        return None
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,start_time,nb_frames",
        "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=timeout)
        stream = json.loads(result.stdout)["streams"][0]
        average = Fraction(stream["avg_frame_rate"])
        nominal = Fraction(stream["r_frame_rate"])
        frame_count = int(stream["nb_frames"])
        start_ms = round(float(stream.get("start_time") or 0) * 1000)
    except (FileNotFoundError, subprocess.TimeoutExpired, KeyError, IndexError,
            TypeError, ValueError, json.JSONDecodeError, ZeroDivisionError):
        return None
    if result.returncode != 0 or average <= 0 or average != nominal or frame_count <= 0:
        return None
    by_index: dict[int, tuple[int, int, int]] = {}
    for requested_ms in sorted(set(requested_points)):
        relative_ms = max(0, requested_ms - start_ms)
        frame_index = math.ceil(relative_ms * average.numerator
                                / (1000 * average.denominator))
        if frame_index >= frame_count:
            continue
        pts_ms = start_ms + round(frame_index * 1000 * average.denominator
                                  / average.numerator)
        candidate = (frame_index, requested_ms, pts_ms)
        current = by_index.get(frame_index)
        if current is None or abs(pts_ms - requested_ms) < abs(pts_ms - current[1]):
            by_index[frame_index] = candidate
    return sorted(by_index.values())


def decode_frames(
    video_path: Path, requested_points: Iterable[int], timeout: int = 900,
) -> Iterator[DecodedFrame]:
    """Decode every requested frame with one FFmpeg process and bounded memory."""
    points = list(requested_points)
    targets = _cfr_targets(video_path, points)
    if targets is None:
        targets = _map_requests_to_frames(points, probe_frame_pts(video_path))
    if not targets:
        return
    select_expression = _balanced_select_expression(
        [frame_index for frame_index, _, _ in targets]
    )
    with tempfile.TemporaryDirectory(prefix="ocr-frames-") as temp_dir:
        pattern = str(Path(temp_dir) / "frame-%06d.png")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-map", "0:v:0", "-vf", f"select={select_expression}",
            "-fps_mode", "passthrough", "-start_number", "0", pattern,
        ]
        try:
            result = subprocess.run(command, capture_output=True, check=False, timeout=timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("FFmpeg 批量解码不可用或执行超时") from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg 批量解码失败: {detail}")
        files = sorted(Path(temp_dir).glob("frame-*.png"))
        if len(files) != len(targets):
            raise RuntimeError(
                f"FFmpeg 批量解码帧数不一致: expected={len(targets)}, actual={len(files)}"
            )
        for path, (_, requested_ms, pts_ms) in zip(files, targets):
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.load()
            except (UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(f"FFmpeg 输出不是有效图片: {path.name}") from exc
            yield DecodedFrame(requested_ms, pts_ms, image)


def _bbox_from_points(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    points = list(points)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, top = min(xs), min(ys)
    right, bottom = max(xs), max(ys)
    return left, top, right - left, bottom - top


def _full_frame_box(item: OCRBox, crop: tuple[int, int, int, int],
                    frame_size: tuple[int, int]) -> SubtitleBox:
    x0, y0, _, _ = crop
    width, height = frame_size
    points = tuple(((x0 + x) / width, (y0 + y) / height) for x, y in item.points)
    left, top, box_width, box_height = _bbox_from_points(points)
    left = max(0.0, min(1.0, left))
    top = max(0.0, min(1.0, top))
    right = max(left, min(1.0, left + box_width))
    bottom = max(top, min(1.0, top + box_height))
    return SubtitleBox(item.text.strip(), item.score, (left, top, right - left, bottom - top))


def _merged_bbox(boxes: list[SubtitleBox]) -> tuple[float, float, float, float]:
    left = min(box.bbox[0] for box in boxes)
    top = min(box.bbox[1] for box in boxes)
    right = max(box.bbox[0] + box.bbox[2] for box in boxes)
    bottom = max(box.bbox[1] + box.bbox[3] for box in boxes)
    return left, top, right - left, bottom - top


def spatial_subtitle_mask(
    array: np.ndarray,
    crop: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    center_band: tuple[float, float],
    padding: float = 0.04,
) -> np.ndarray:
    """Black out pixels outside the subtitle band before text detection."""
    _, crop_y0, _, crop_y1 = crop
    _, frame_height = frame_size
    keep_y0 = round(frame_height * max(0.0, center_band[0] - padding)) - crop_y0
    keep_y1 = round(frame_height * min(1.0, center_band[1] + padding)) - crop_y0
    keep_y0 = max(0, min(array.shape[0], keep_y0))
    keep_y1 = max(keep_y0, min(array.shape[0], keep_y1))
    masked = np.zeros_like(array)
    masked[keep_y0:keep_y1] = array[keep_y0:keep_y1]
    return masked


def _subtitle_boxes(
    raw_boxes: list[OCRBox], crop_box: tuple[int, int, int, int],
    frame_size: tuple[int, int], settings: Settings,
) -> list[SubtitleBox]:
    boxes = [
        _full_frame_box(item, crop_box, frame_size)
        for item in raw_boxes if item.text.strip()
    ]
    band_start, band_end = settings.subtitle_center_band
    boxes = [box for box in boxes if band_start <= box.center[1] <= band_end]
    boxes.sort(key=lambda box: (round(box.center[1], 2), box.center[0]))
    return boxes


def ocr_subtitle_frame(image: Image.Image, requested_ms: int, pts_ms: int,
                       provider: RapidOCRProvider, settings: Settings) -> FrameOCR:
    width, height = image.size
    x0 = round(width * settings.roi[0])
    y0 = round(height * settings.roi[1])
    x1 = round(width * settings.roi[2])
    y1 = round(height * settings.roi[3])
    crop_box = (x0, y0, x1, y1)
    cropped = image.crop(crop_box)
    array = np.asarray(cropped)
    masked = spatial_subtitle_mask(
        array, crop_box, image.size, settings.subtitle_center_band,
    )
    if settings.background_suppression == "off":
        variants = [("original", array)]
    elif settings.background_suppression == "spatial":
        variants = [("spatial", masked)]
    else:
        variants = [("original", array), ("spatial", masked)]
    started = time.perf_counter()
    recognized = [
        (name, _subtitle_boxes(provider.recognize(variant), crop_box, image.size, settings))
        for name, variant in variants
    ]
    elapsed = (time.perf_counter() - started) * 1000
    original_boxes = next((boxes for name, boxes in recognized if name == "original"), [])
    spatial_boxes = next((boxes for name, boxes in recognized if name == "spatial"), [])
    if settings.background_suppression == "adaptive" and spatial_boxes:
        original_score = sum(box.score for box in original_boxes) / len(original_boxes) \
            if original_boxes else 0.0
        spatial_score = sum(box.score for box in spatial_boxes) / len(spatial_boxes)
        boxes = spatial_boxes if spatial_score >= original_score - 0.05 else original_boxes
    else:
        boxes = recognized[0][1]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0
    return FrameOCR(
        requested_ms=requested_ms,
        pts_ms=pts_ms,
        pts_guard_ok=abs(pts_ms - requested_ms) <= settings.decode_pts_guard_ms,
        text="".join(box.text for box in boxes),
        confidence=sum(box.score for box in boxes) / len(boxes) if boxes else 0.0,
        bbox=_merged_bbox(boxes) if boxes else None,
        boxes=boxes,
        sharpness=sharpness,
        ocr_ms=elapsed,
    )


def _near(first: SubtitleBox, second: SubtitleBox, tolerance: float = 0.04) -> bool:
    ax, ay = first.center
    bx, by = second.center
    return abs(ax - bx) <= tolerance and abs(ay - by) <= tolerance


def _box_tracks(frames: list[FrameOCR]) -> list[list[tuple[FrameOCR, SubtitleBox]]]:
    tracks: list[list[tuple[FrameOCR, SubtitleBox]]] = []
    for frame in sorted(frames, key=lambda row: row.pts_ms):
        for box in frame.boxes:
            matching = next((track for track in tracks
                             if track[-1][1].text == box.text and _near(track[-1][1], box)), None)
            if matching is None:
                tracks.append([(frame, box)])
            else:
                matching.append((frame, box))
    return tracks


def _best_frame(frames: list[FrameOCR], midpoint: float) -> FrameOCR:
    return max(frames, key=lambda row: (
        row.confidence, row.sharpness, -abs(row.pts_ms - midpoint), -row.pts_ms,
    ))


def select_segment(segment: dict[str, Any], frames: list[FrameOCR],
                   settings: Settings) -> tuple[FrameOCR, str, tuple[float, float, float, float], float, dict[str, Any]] | None:
    start, end = int(segment["start_ms"]), int(segment["end_ms"])
    inner_start, inner_end = cue_window(segment, settings)
    fallback = [row for row in frames if start <= row.pts_ms <= end and row.text]
    inner = [row for row in fallback
             if inner_start <= row.pts_ms <= inner_end and row.pts_guard_ok]
    stage = "trimmed"
    active = inner
    if not active:
        active = fallback
        stage = "untrimmed_fallback"
    if not active:
        return None
    midpoint = (start + end) / 2
    confirmed = []
    for track in _box_tracks(active):
        span = track[-1][0].pts_ms - track[0][0].pts_ms
        text = track[0][1].text
        if len(track) >= 2 and (len(text) > 1 or span >= settings.single_char_confirmation_gap_ms):
            avg_score = sum(item.score for _, item in track) / len(track)
            confirmed.append((len(track), span, avg_score, track))
    if confirmed:
        _, span, _, track = max(confirmed, key=lambda item: (item[0], item[1], item[2]))
        frame, box = max(track, key=lambda item: (
            item[0].sharpness, item[1].score, -abs(item[0].pts_ms - midpoint),
        ))
        return frame, box.text, box.bbox, box.score, {
            "stage": stage,
            "evidence": "confirmed_box_track",
            "candidate_count": len(active),
            "evidence_span_ms": span,
        }
    frame = _best_frame(active, midpoint)
    evidence = "single_frame_fallback" if len(active) == 1 else "visible_fallback"
    return frame, frame.text, frame.bbox, frame.confidence, {
        "stage": stage,
        "evidence": evidence,
        "candidate_count": len(active),
        "evidence_span_ms": 0,
    }


def detection_for_segment(index: int, segment: dict[str, Any], frames: list[FrameOCR],
                          provider: RapidOCRProvider, settings: Settings) -> dict[str, Any] | None:
    selected = select_segment(segment, frames, settings)
    if selected is None:
        return None
    frame, text, bbox, confidence, meta = selected
    if not text or bbox is None:
        return None
    status = provider.status
    version = f"rapidocr@{status.rapidocr_version or 'unknown'}/{status.model}/{status.actual_provider or 'unknown'}"
    return {
        "id": f"segment-{index + 1:06d}",
        "segment_index": index,
        "requested_ms": frame.requested_ms,
        "frame_pts_ms": frame.pts_ms,
        "bbox": [round(value, 6) for value in bbox],
        "ocr_text": text,
        "confidence": round(confidence, 4),
        "engine_version": version,
        "kind": "subtitle",
        "asr_text_prompted": False,
        "selection": meta,
    }


def process_video(video_path: Path, segments: list[dict[str, Any]],
                  provider: RapidOCRProvider, settings: Settings) -> list[dict[str, Any]]:
    duration_ms = probe_duration_ms(video_path)
    points = candidate_points(duration_ms, segments, settings)
    frames: list[FrameOCR] = []
    ocr_cache: dict[int, FrameOCR] = {}
    try:
        decoded_frames = decode_frames(video_path, points)
        for decoded in decoded_frames:
            image = decoded.image
            requested_ms = decoded.requested_ms
            pts_ms = decoded.pts_ms
            cached = ocr_cache.get(pts_ms)
            if cached is None:
                cached = ocr_subtitle_frame(image, requested_ms, pts_ms, provider, settings)
                ocr_cache[pts_ms] = cached
                frames.append(cached)
            elif abs(pts_ms - requested_ms) < abs(cached.pts_ms - cached.requested_ms):
                cached.requested_ms = requested_ms
                cached.pts_guard_ok = abs(pts_ms - requested_ms) <= settings.decode_pts_guard_ms
    except RuntimeError as exc:
        raise RuntimeError(f"FFmpeg 无法批量解码候选帧: {exc}") from exc
    if not frames:
        raise RuntimeError("FFmpeg 无法解码任何候选帧")
    detections = []
    for index, segment in enumerate(segments):
        detection = detection_for_segment(index, segment, frames, provider, settings)
        if detection:
            detections.append(detection)
    return detections
