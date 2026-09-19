import hashlib
import json
from pathlib import Path

import pytest

import app.asr_adapter as adapter
from app.asr_adapter import load_asr_input, run_asr_transcript, save_evidence
from app.settings import Settings


class FakeProvider:
    def __init__(self):
        self.ready = False

    def ensure_ready(self):
        self.ready = True
        return self


def write_asr_job(tmp_path: Path, *, source_file: str = "episode.mp4") -> tuple[Path, Path, str]:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"matching-video")
    media_id = hashlib.sha256(video.read_bytes()).hexdigest()
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({
        "schema_version": "1.0",
        "media_id": media_id,
        "source_file": source_file,
        "segments": [
            {"segment_id": "s000001", "start_ms": 1000, "end_ms": 1300,
             "asr_text": "不会传给 OCR"},
        ],
    }), encoding="utf-8")
    return transcript, video, media_id


def test_load_asr_input_resolves_relative_source_and_verifies_hash(tmp_path: Path):
    transcript, video, media_id = write_asr_job(tmp_path)

    job = load_asr_input(transcript)

    assert job.video_path == video.resolve()
    assert job.media_id == media_id
    assert job.segments == [{"start_ms": 1000, "end_ms": 1300}]


def test_load_asr_input_rejects_media_mismatch(tmp_path: Path):
    transcript, _, _ = write_asr_job(tmp_path)
    document = json.loads(transcript.read_text(encoding="utf-8"))
    document["media_id"] = "0" * 64
    transcript.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="不一致"):
        load_asr_input(transcript)


def test_load_asr_input_rejects_non_object_json(tmp_path: Path):
    transcript = tmp_path / "transcript.json"
    transcript.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="顶层"):
        load_asr_input(transcript)


def test_run_asr_transcript_returns_asr_compatible_evidence(monkeypatch, tmp_path: Path):
    transcript, video, media_id = write_asr_job(tmp_path)
    provider = FakeProvider()
    captured = {}

    def fake_process(path, segments, used_provider, settings):
        captured.update(path=path, segments=segments, provider=used_provider, settings=settings)
        return [{"id": "segment-000001", "frame_pts_ms": 1100}]

    monkeypatch.setattr(adapter, "process_video", fake_process)
    settings = Settings(execution_mode="cpu")

    result = run_asr_transcript(transcript, provider=provider, settings=settings)

    assert provider.ready is True
    assert result == {
        "schema_version": "1.0",
        "media_id": media_id,
        "timebase": "video_ms",
        "detections": [{"id": "segment-000001", "frame_pts_ms": 1100}],
    }
    assert captured["path"] == video.resolve()
    assert captured["segments"] == [{"start_ms": 1000, "end_ms": 1300}]
    assert captured["provider"] is provider
    assert captured["settings"] is settings


def test_video_override_is_still_checked_against_asr_media_id(tmp_path: Path):
    transcript, _, _ = write_asr_job(tmp_path, source_file="missing.mp4")
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"another-video")

    with pytest.raises(ValueError, match="不一致"):
        load_asr_input(transcript, replacement)


def test_save_evidence_writes_utf8_json_atomically(tmp_path: Path):
    path = tmp_path / "nested" / "ocr_evidence.json"
    evidence = {"media_id": "a" * 64, "detections": [{"ocr_text": "陆总"}]}

    assert save_evidence(path, evidence) == path.resolve()
    assert json.loads(path.read_text(encoding="utf-8")) == evidence
    assert not list(path.parent.glob("*.tmp"))
