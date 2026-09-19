from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import MAX_TRANSCRIPT_BYTES, SCHEMA_VERSION, validate_segments
from .ocr_engine import RapidOCRProvider
from .settings import Settings
from .video_pipeline import process_video


@dataclass(frozen=True)
class ASRInput:
    transcript_path: Path
    video_path: Path
    media_id: str
    segments: list[dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_asr_input(transcript_path: Path, video_path: Path | None = None) -> ASRInput:
    """Load one ASR transcript and resolve the matching source video."""
    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"ASR transcript 不存在: {transcript_path}")
    if transcript_path.stat().st_size > MAX_TRANSCRIPT_BYTES:
        raise ValueError("transcript 文件过大")

    try:
        document = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法解析 ASR transcript: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("ASR transcript 顶层必须是 JSON 对象")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"ASR schema_version 必须是 {SCHEMA_VERSION}")
    segments = validate_segments(document)

    media_id = document.get("media_id")
    if not isinstance(media_id, str) or len(media_id) != 64:
        raise ValueError("ASR media_id 必须是视频文件的 SHA-256")
    try:
        int(media_id, 16)
    except ValueError as exc:
        raise ValueError("ASR media_id 必须是视频文件的 SHA-256") from exc
    media_id = media_id.lower()

    selected_video: Path
    if video_path is not None:
        selected_video = video_path.expanduser()
    else:
        source_file = document.get("source_file")
        if not isinstance(source_file, str) or not source_file.strip():
            raise ValueError("ASR transcript 缺少 source_file")
        selected_video = Path(source_file).expanduser()
    if not selected_video.is_absolute():
        selected_video = transcript_path.parent / selected_video
    selected_video = selected_video.resolve()
    if not selected_video.is_file():
        raise FileNotFoundError(f"ASR 源视频不存在: {selected_video}")

    actual_media_id = sha256_file(selected_video)
    if not hmac.compare_digest(media_id, actual_media_id):
        raise ValueError(
            "ASR media_id 与源视频 SHA-256 不一致: "
            f"transcript={media_id}, video={actual_media_id}"
        )
    return ASRInput(transcript_path, selected_video, media_id, segments)


def run_asr_transcript(
    transcript_path: Path,
    *,
    video_path: Path | None = None,
    provider: RapidOCRProvider | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run OCR directly from files emitted by ASR."""
    job = load_asr_input(transcript_path, video_path)
    active_settings = settings or Settings.from_env()
    active_provider = provider or RapidOCRProvider(active_settings)
    active_provider.ensure_ready()
    detections = process_video(job.video_path, job.segments, active_provider, active_settings)
    return {
        "schema_version": SCHEMA_VERSION,
        "media_id": job.media_id,
        "timebase": "video_ms",
        "detections": detections,
    }


def save_evidence(path: Path, evidence: dict[str, Any]) -> Path:
    """Atomically save UTF-8 OCR evidence for ASR ocr-propose."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as target:
            json.dump(evidence, target, ensure_ascii=False, indent=2)
            target.write("\n")
            temporary = Path(target.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
