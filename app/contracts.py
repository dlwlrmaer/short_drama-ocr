from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.0"
MAX_TRANSCRIPT_BYTES = 10 * 1024 * 1024
MAX_SEGMENTS = 20_000
MAX_BATCH_IMAGES = 64


def validate_segments(document: Any) -> list[dict[str, Any]]:
    """Validate the shared ASR/OCR timing contract without reading ASR text."""
    if not isinstance(document, dict) or not isinstance(document.get("segments"), list):
        raise ValueError("transcript 必须包含 segments 数组")
    if len(document["segments"]) > MAX_SEGMENTS:
        raise ValueError("segments 数量超过限制")
    validated = []
    for index, segment in enumerate(document["segments"]):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{index}] 必须是对象")
        start, end = segment.get("start_ms"), segment.get("end_ms")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or start < 0
            or end <= start
        ):
            raise ValueError(f"segments[{index}] 时间区间无效")
        validated.append({"start_ms": int(start), "end_ms": int(end)})
    return validated
