"""Run local health, image OCR and generated-video API smoke checks."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    client = TestClient(app)
    health = client.get("/health")
    with args.image.open("rb") as source:
        image_response = client.post(
            "/ocr", files={"file": (args.image.name, source.read(), "image/png")})
    with tempfile.TemporaryDirectory(prefix="ocr-smoke-") as directory:
        video = Path(directory) / "sample.mp4"
        generated = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=c=black:s=320x240:d=1:r=25", "-c:v", "libx264", str(video),
        ], capture_output=True, check=False)
        if generated.returncode != 0:
            raise RuntimeError(generated.stderr.decode(errors="replace"))
        transcript = json.dumps({"segments": [{"start_ms": 0, "end_ms": 800}]})
        video_response = client.post("/ocr/video", files={
            "video": (video.name, video.read_bytes(), "video/mp4"),
            "transcript": ("transcript.json", transcript.encode(), "application/json"),
        })
    report = {
        "health_status": health.status_code,
        "health": health.json(),
        "image_status": image_response.status_code,
        "image_text": image_response.json().get("text") if image_response.status_code == 200 else None,
        "video_status": video_response.status_code,
        "video_detection_count": len(video_response.json().get("detections", []))
        if video_response.status_code == 200 else None,
        "video_error": video_response.json().get("detail")
        if video_response.status_code != 200 else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if health.status_code != 200 or image_response.status_code != 200 or video_response.status_code != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
