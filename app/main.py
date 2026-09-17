import hashlib
import json
import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="NAS OCR API", version="1.0.0")

FRAME_PADDING_MS = 500
FRAME_INTERVAL_MS = 1000
MAX_TRANSCRIPT_BYTES = 10 * 1024 * 1024
MAX_SEGMENTS = 20_000
PTS_RE = re.compile(r"pts_time:([-+]?\d+(?:\.\d+)?)")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="请上传图片文件")

    try:
        image = Image.open(BytesIO(await file.read()))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="无法解析图片") from exc

    text = pytesseract.image_to_string(image, lang="chi_sim+eng")
    return {"filename": file.filename or "image", "text": text.strip()}


def validate_segments(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict) or not isinstance(document.get("segments"), list):
        raise ValueError("transcript 必须包含 segments 数组")
    if len(document["segments"]) > MAX_SEGMENTS:
        raise ValueError("segments 数量超过限制")
    validated = []
    for index, segment in enumerate(document["segments"]):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{index}] 必须是对象")
        start, end = segment.get("start_ms"), segment.get("end_ms")
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or
                start < 0 or end <= start):
            raise ValueError(f"segments[{index}] 时间区间无效")
        validated.append({"start_ms": int(start), "end_ms": int(end)})
    return validated


def sample_points(segments: list[dict[str, Any]], duration_ms: int | None = None) -> list[int]:
    points: set[int] = set()
    for segment in segments:
        start = max(0, segment["start_ms"] - FRAME_PADDING_MS)
        end = segment["end_ms"] + FRAME_PADDING_MS
        if duration_ms is not None:
            end = min(end, duration_ms)
        points.update(range(start, end + 1, FRAME_INTERVAL_MS))
        points.add(start)
        points.add(end)
    return sorted(points)


def decode_frame(video_path: Path, requested_ms: int) -> tuple[bytes, int]:
    requested_seconds = f"{requested_ms / 1000:.3f}"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(video_path),
               "-frames:v", "1", "-vf", f"select=gte(t\\,{requested_seconds}),showinfo",
               "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("FFmpeg 不可用或执行超时") from exc
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError("FFmpeg 无法解码请求帧")
    match = PTS_RE.search(result.stderr.decode("utf-8", errors="replace"))
    if not match:
        raise RuntimeError("FFmpeg 未返回真实帧 PTS")
    actual_ms = round(float(match.group(1)) * 1000)
    return result.stdout, actual_ms


def image_detection(image_bytes: bytes, frame_pts_ms: int, detection_id: str) -> dict[str, Any] | None:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError("FFmpeg 输出不是有效图片") from exc
    data = pytesseract.image_to_data(image, lang="chi_sim+eng", output_type=pytesseract.Output.DICT)
    width, height = image.size
    words, boxes, confidences = [], [], []
    for text, left, top, box_width, box_height, confidence in zip(
            data.get("text", []), data.get("left", []), data.get("top", []),
            data.get("width", []), data.get("height", []), data.get("conf", [])):
        text = text.strip()
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if text and confidence >= 0:
            words.append(text)
            boxes.append((int(left), int(top), int(box_width), int(box_height)))
            confidences.append(confidence / 100)
    if not words:
        return None
    x = min(item[0] for item in boxes)
    y = min(item[1] for item in boxes)
    right = max(item[0] + item[2] for item in boxes)
    bottom = max(item[1] + item[3] for item in boxes)
    return {"id": detection_id, "frame_pts_ms": frame_pts_ms,
            "bbox": [x / width, y / height, (right - x) / width, (bottom - y) / height],
            "ocr_text": "".join(words), "confidence": round(sum(confidences) / len(confidences), 3),
            "engine_version": f"pytesseract@{getattr(pytesseract, '__version__', 'unknown')}",
            "kind": "subtitle", "asr_text_prompted": False}


@app.post("/ocr/video")
async def video_ocr(video: UploadFile = File(...), transcript: UploadFile = File(...)) -> dict[str, Any]:
    if not video.filename or not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="请上传视频文件")
    transcript_bytes = await transcript.read(MAX_TRANSCRIPT_BYTES + 1)
    if len(transcript_bytes) > MAX_TRANSCRIPT_BYTES:
        raise HTTPException(status_code=413, detail="transcript 文件过大")
    try:
        document = json.loads(transcript_bytes.decode("utf-8"))
        segments = validate_segments(document)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with tempfile.TemporaryDirectory(prefix="ocr-video-") as temp_dir:
        video_path = Path(temp_dir) / "input-video"
        digest = hashlib.sha256()
        with video_path.open("wb") as target:
            while chunk := await video.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        detections = []
        try:
            for index, requested_ms in enumerate(sample_points(segments)):
                frame, actual_ms = decode_frame(video_path, requested_ms)
                detection = image_detection(frame, actual_ms, f"frame-{index + 1:06d}")
                if detection:
                    detections.append(detection)
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"media_id": digest.hexdigest(), "timebase": "video_ms", "detections": detections}
