import hashlib
import json
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from .contracts import MAX_TRANSCRIPT_BYTES, validate_segments
from .ocr_engine import get_provider
from .settings import Settings
from .video_pipeline import process_video


app = FastAPI(title="NAS OCR API", version="1.2.0")

FRAME_PADDING_MS = 500
FRAME_INTERVAL_MS = 1000


@app.get("/health")
def health():
    provider = get_provider()
    try:
        provider.ensure_ready()
        ocr_ready = True
    except Exception:
        ocr_ready = False
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    video_ready = ocr_ready and bool(ffmpeg_path and ffprobe_path)
    content = {
        "status": "ok" if video_ready else "degraded",
        "ocr": provider.status.to_dict(),
        "ffmpeg": {"ready": bool(ffmpeg_path and ffprobe_path),
                   "ffmpeg": ffmpeg_path, "ffprobe": ffprobe_path},
        "video_ocr_ready": video_ready,
    }
    return JSONResponse(status_code=200 if video_ready else 503, content=content)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="请上传图片文件")

    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="无法解析图片") from exc

    try:
        text = get_provider().text(np.asarray(image))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"OCR 引擎不可用: {exc}") from exc
    return {"filename": file.filename or "image", "text": text.strip()}


def sample_points(segments: list[dict[str, Any]], duration_ms: int | None = None) -> list[int]:
    """保留旧的稀疏抽帧辅助函数，供既有调用和回归测试使用。"""
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
        try:
            provider = get_provider()
            provider.ensure_ready()
            detections = process_video(video_path, segments, provider, Settings.from_env())
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"OCR 引擎不可用: {exc}") from exc
    return {"media_id": digest.hexdigest(), "timebase": "video_ms", "detections": detections}
