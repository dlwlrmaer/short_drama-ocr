import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO

from app.main import app, sample_points, validate_segments


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
    monkeypatch.setattr("app.main.pytesseract.image_to_string", lambda image, lang: "字幕")
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_bytes, format="PNG")
    response = TestClient(app).post("/ocr", files={"file": ("caption.png", image_bytes.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"filename": "caption.png", "text": "字幕"}
