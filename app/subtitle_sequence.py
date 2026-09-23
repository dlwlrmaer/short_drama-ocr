"""Recover a sequence of visible subtitles using ASR as a sampling hint.

The ASR text is never passed to OCR. Every sampled frame can contribute a
subtitle, including frames outside all ASR windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .ocr_engine import RapidOCRProvider
from .settings import Settings
from .video_pipeline import FrameOCR, decode_frames, ocr_subtitle_frame, probe_duration_ms


@dataclass
class _Run:
    observations: list[FrameOCR] = field(default_factory=list)

    @property
    def first_ms(self) -> int:
        return self.observations[0].pts_ms

    @property
    def last_ms(self) -> int:
        return self.observations[-1].pts_ms

    @property
    def best(self) -> FrameOCR:
        return max(self.observations, key=lambda row: (
            len(row.text.strip()), row.confidence, row.sharpness,
        ))


def sample_points(duration_ms: int, segments: list[dict[str, Any]],
                  speech_ms: int, fallback_ms: int, padding_ms: int) -> list[int]:
    """Sample the full video; add denser points near independent speech timing."""
    if speech_ms <= 0 or fallback_ms <= 0 or padding_ms < 0:
        raise ValueError("采样间隔必须大于 0，padding 不能为负")
    points = set(range(0, duration_ms, fallback_ms))
    for segment in segments:
        start = max(0, int(segment["start_ms"]) - padding_ms)
        end = min(duration_ms, int(segment["end_ms"]) + padding_ms)
        points.update(range(start, end, speech_ms))
        points.add(end)
    return sorted(point for point in points if point < duration_ms)


def _same_text(left: str, right: str) -> bool:
    left, right = left.strip(), right.strip()
    if left == right:
        return True
    if min(len(left), len(right)) < 3:
        return False
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.82


def normalize_ocr_text(value: str) -> str:
    """Apply the requested drama-specific censorship character correction."""
    return value.strip().replace("亖", "死").replace("三", "死")


def is_background_noise(text: str, confidence: float, observations: int) -> bool:
    """Reject brief OCR hallucinations from patterned clothes and set dressing."""
    chars = [char for char in text if char.isalnum()]
    if not chars or observations >= 3:
        return False
    han = sum("\u4e00" <= char <= "\u9fff" for char in chars)
    digits = sum(char.isdigit() for char in chars)
    latin = sum(char.isascii() and char.isalpha() for char in chars)
    if han == 0:
        return bool((digits and latin) or digits == len(chars) or confidence < 0.9)
    return (han / len(chars) < 0.5 and latin + digits >= 3
            and confidence < 0.9)


def _group_frames(frames: list[FrameOCR], max_gap_ms: int,
                  blank_break_ms: int) -> list[_Run]:
    runs: list[_Run] = []
    active: _Run | None = None
    blank_seen = False
    for row in frames:
        if not row.text.strip():
            blank_seen = True
            continue
        if (active is not None and row.pts_ms - active.last_ms <= max_gap_ms
                and _same_text(active.best.text, row.text)
                and not (blank_seen and row.pts_ms - active.last_ms >= blank_break_ms
                         and len(active.observations) >= 2)):
            active.observations.append(row)
        else:
            active = _Run([row])
            runs.append(active)
        blank_seen = False
    # One OCR dropout can briefly split a continuous caption. Require the new
    # appearance to be confirmed by at least two frames before keeping it.
    merged: list[_Run] = []
    for run in runs:
        if (merged and (len(merged[-1].observations) == 1 or len(run.observations) == 1)
                and run.first_ms - merged[-1].last_ms <= max_gap_ms
                and _same_text(merged[-1].best.text, run.best.text)):
            merged[-1].observations.extend(run.observations)
        else:
            merged.append(run)
    return merged


def _speech_overlap(start_ms: int, end_ms: int,
                    segments: list[dict[str, Any]]) -> bool:
    return any(int(segment["start_ms"]) <= end_ms
               and int(segment["end_ms"]) >= start_ms for segment in segments)


def sequence_detections(frames: list[FrameOCR], segments: list[dict[str, Any]],
                        duration_ms: int, speech_ms: int, fallback_ms: int,
                        engine_version: str = "rapidocr"
                        ) -> list[dict[str, Any]]:
    """Keep every distinct visual subtitle track, with estimated screen times."""
    frames = sorted(frames, key=lambda row: row.pts_ms)
    runs = _group_frames(frames, max_gap_ms=max(2 * fallback_ms, 2 * speech_ms),
                         blank_break_ms=max(400, min(speech_ms, fallback_ms)))
    detections: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        best = run.best
        before = max((row.pts_ms for row in frames if row.pts_ms < run.first_ms),
                     default=max(0, run.first_ms - fallback_ms))
        after = min((row.pts_ms for row in frames if row.pts_ms > run.last_ms),
                    default=min(duration_ms, run.last_ms + fallback_ms))
        start_ms = max(0, (before + run.first_ms) // 2)
        end_ms = min(duration_ms, (run.last_ms + after) // 2)
        guided = _speech_overlap(start_ms, end_ms, segments)
        # A lone short background mark is often an overlay, not dialogue.
        if len(run.observations) == 1 and not guided and len(best.text.strip()) <= 2:
            continue
        if best.bbox is None or not best.text.strip() or end_ms <= start_ms:
            continue
        # Near-zero-height marks and full-ROI short tokens are detector noise.
        if best.bbox[3] < 0.02 or (best.bbox[3] >= 0.13 and len(best.text.strip()) <= 3):
            continue
        if is_background_noise(best.text, best.confidence, len(run.observations)):
            continue
        if detections and detections[-1]["end_ms"] > start_ms:
            detections[-1]["end_ms"] = start_ms
        detections.append({
            "id": f"visual-{len(detections) + 1:06d}",
            "start_ms": start_ms, "end_ms": end_ms,
            "frame_pts_ms": best.pts_ms,
            "ocr_text": normalize_ocr_text(best.text),
            "raw_ocr_text": best.text.strip(),
            "confidence": round(best.confidence, 4),
            "bbox": [round(value, 6) for value in best.bbox],
            "observations": len(run.observations),
            "guidance": "asr_timing" if guided else "visual_fallback",
            "asr_text_prompted": False,
            "kind": "subtitle",
            "engine_version": engine_version,
        })
    return detections


def process_video_sequence(video_path: Path, segments: list[dict[str, Any]],
                           provider: RapidOCRProvider, settings: Settings,
                           *, speech_ms: int | None = None,
                           fallback_ms: int | None = None,
                           chunk_points: int = 1200) -> list[dict[str, Any]]:
    """Scan one video in bounded decoding batches and return visible cues."""
    duration_ms = probe_duration_ms(video_path)
    speech_ms = speech_ms or (250 if settings.runtime_profile == "win11" else 450)
    fallback_ms = fallback_ms or (450 if settings.runtime_profile == "win11" else 700)
    points = sample_points(duration_ms, segments, speech_ms, fallback_ms,
                           settings.padding_ms)
    frames: list[FrameOCR] = []
    for offset in range(0, len(points), chunk_points):
        for decoded in decode_frames(video_path, points[offset:offset + chunk_points]):
            frames.append(ocr_subtitle_frame(decoded.image, decoded.requested_ms,
                                             decoded.pts_ms, provider, settings))
    if not frames:
        raise RuntimeError("视频没有可解码的采样帧")
    status = provider.status
    engine_version = f"{status.model}@{status.rapidocr_version or 'unknown'}"
    return sequence_detections(frames, segments, duration_ms, speech_ms,
                               fallback_ms, engine_version)
