import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO

import app.main as main_module
from app.main import app, sample_points, validate_segments


class FakeStatus:
    def to_dict(self):
        return {"ready": True, "requested_mode": "cpu",
                "actual_provider": "CPUExecutionProvider"}


class FakeProvider:
    status = FakeStatus()

    def __init__(self):
        self.ready_calls = 0
        self.text_calls = 0
        self.settings = SimpleNamespace(max_batch_images=32)

    def ensure_ready(self):
        self.ready_calls += 1
        return self

    def text(self, _image):
        self.text_calls += 1
        return "字幕"


def test_sample_points_expands_segments_and_deduplicates():
    assert sample_points([{"start_ms": 1000, "end_ms": 1200}]) == [500, 1500, 1700]
    points = sample_points([{"start_ms": 0, "end_ms": 500}, {"start_ms": 400, "end_ms": 1400}])
    assert points == sorted(set(points))
    assert 0 in points and 1900 in points


def test_validate_segments_rejects_invalid_input():
    with pytest.raises(ValueError):
        validate_segments({})
    with pytest.raises(ValueError):
        validate_segments({"segments": [{"start_ms": 3, "end_ms": 3}]})
    with pytest.raises(ValueError):
        validate_segments({"segments": [{"start_ms": -1, "end_ms": 3}]})


def test_media_id_is_sha256(tmp_path: Path):
    payload = b"video"
    path = tmp_path / "video.mp4"
    path.write_bytes(payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()


def test_existing_image_endpoint_remains_compatible(monkeypatch):
    monkeypatch.setattr(main_module, "get_provider", lambda: FakeProvider())
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_bytes, format="PNG")
    response = TestClient(app).post("/ocr", files={"file": ("caption.png", image_bytes.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"filename": "caption.png", "text": "字幕"}


def test_image_endpoint_returns_empty_text_when_no_boxes(monkeypatch):
    provider = FakeProvider()
    provider.text = lambda _image: ""
    monkeypatch.setattr(main_module, "get_provider", lambda: provider)
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_bytes, format="PNG")
    response = TestClient(app).post(
        "/ocr", files={"file": ("blank.png", image_bytes.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["text"] == ""


def test_batch_image_endpoint_preserves_order_and_reuses_provider(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(main_module, "get_provider", lambda: provider)
    first = BytesIO()
    second = BytesIO()
    Image.new("RGB", (2, 2), "white").save(first, format="PNG")
    Image.new("RGB", (3, 3), "black").save(second, format="PNG")

    response = TestClient(app).post("/ocr/batch", files=[
        ("files", ("first.png", first.getvalue(), "image/png")),
        ("files", ("second.png", second.getvalue(), "image/png")),
    ])

    assert response.status_code == 200
    assert response.json() == {
        "count": 2,
        "results": [
            {"index": 0, "filename": "first.png", "text": "字幕"},
            {"index": 1, "filename": "second.png", "text": "字幕"},
        ],
    }
    assert provider.ready_calls == 1
    assert provider.text_calls == 2


def test_batch_image_endpoint_rejects_invalid_item_and_oversized_batch(monkeypatch):
    monkeypatch.setattr(main_module, "get_provider", lambda: FakeProvider())
    client = TestClient(app)
    invalid = client.post("/ocr/batch", files=[
        ("files", ("note.txt", b"not-image", "text/plain")),
    ])
    assert invalid.status_code == 415
    assert "files[0]" in invalid.json()["detail"]

    too_many = client.post("/ocr/batch", files=[
        ("files", (f"{index}.png", b"ignored", "image/png"))
        for index in range(main_module.MAX_BATCH_IMAGES + 1)
    ])
    assert too_many.status_code == 413


def test_video_endpoint_preserves_contract_and_ignores_asr_text(monkeypatch):
    monkeypatch.setattr(main_module, "get_provider", lambda: FakeProvider())
    captured = []

    def fake_process(path, segments, provider, settings):
        captured.append(segments)
        return [{"id": "segment-000001", "segment_index": 0,
                 "frame_pts_ms": 1100, "bbox": [0.1, 0.7, 0.2, 0.05],
                 "ocr_text": "妈", "confidence": 0.99,
                 "engine_version": "fake", "kind": "subtitle",
                 "asr_text_prompted": False, "selection": {"evidence": "fake"}}]

    monkeypatch.setattr(main_module, "process_video", fake_process)
    transcript = {"segments": [{"start_ms": 1000, "end_ms": 1300, "text": "错误提示"}]}
    response = TestClient(app).post("/ocr/video?mode=legacy", files={
        "video": ("episode.mp4", b"video", "video/mp4"),
        "transcript": ("transcript.json", json.dumps(transcript).encode(), "application/json"),
    })
    assert response.status_code == 200
    body = response.json()
    assert body["media_id"] == hashlib.sha256(b"video").hexdigest()
    assert body["timebase"] == "video_ms"
    assert body["detections"][0]["asr_text_prompted"] is False
    assert captured == [[{"start_ms": 1000, "end_ms": 1300}]]


def test_video_endpoint_defaults_to_sequence_and_rejects_wrong_media(monkeypatch):
    monkeypatch.setattr(main_module, "get_provider", lambda: FakeProvider())
    captured = []

    def fake_sequence(path, segments, provider, settings):
        captured.append(segments)
        return [{"id": "visual-000001", "start_ms": 100, "end_ms": 500,
                 "frame_pts_ms": 200, "ocr_text": "妈", "kind": "subtitle",
                 "asr_text_prompted": False}]

    monkeypatch.setattr(main_module, "process_video_sequence", fake_sequence)
    client = TestClient(app)
    video = b"video"
    files = {
        "video": ("episode.mp4", video, "video/mp4"),
        "transcript": ("transcript.json", json.dumps({
            "media_id": hashlib.sha256(video).hexdigest(),
            "segments": [{"start_ms": 100, "end_ms": 500, "asr_text": "无关"}],
        }).encode(), "application/json"),
    }
    response = client.post("/ocr/video", files=files)
    assert response.status_code == 200
    assert captured == [[{"start_ms": 100, "end_ms": 500}]]
    assert response.json()["detections"][0]["ocr_text"] == "妈"

    bad = client.post("/ocr/video", files={**files, "transcript": (
        "transcript.json", json.dumps({"media_id": "0" * 64, "segments": []}).encode(),
        "application/json")})
    assert bad.status_code == 400
    assert len(captured) == 1


def test_health_exposes_provider_and_ffmpeg(monkeypatch):
    monkeypatch.setattr(main_module, "get_provider", lambda: FakeProvider())
    monkeypatch.setattr(main_module.shutil, "which", lambda name: f"C:/tools/{name}.exe")
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ocr"]["actual_provider"] == "CPUExecutionProvider"
    assert body["video_ocr_ready"] is True


def test_health_is_degraded_when_ocr_or_ffmpeg_is_unavailable(monkeypatch):
    class FailedProvider(FakeProvider):
        status = FakeStatus()

        def ensure_ready(self):
            self.status.to_dict = lambda: {"ready": False, "error": "CUDA DLL missing"}
            raise RuntimeError("CUDA DLL missing")

    monkeypatch.setattr(main_module, "get_provider", lambda: FailedProvider())
    monkeypatch.setattr(main_module.shutil, "which", lambda _name: None)
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json()["ocr"]["error"] == "CUDA DLL missing"
    assert response.json()["ffmpeg"]["ready"] is False
