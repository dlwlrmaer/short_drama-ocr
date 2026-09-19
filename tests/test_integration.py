import shutil
import subprocess
from pathlib import Path

import pytest

from app.ocr_engine import CUDA_PROVIDER, RapidOCRProvider
from app.settings import Settings
from app.video_pipeline import decode_frame, decode_frames, probe_duration_ms


@pytest.mark.integration
def test_generated_video_has_duration_and_real_pts(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe unavailable")
    video = tmp_path / "sample.mp4"
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=black:s=320x240:d=1:r=25", "-c:v", "libx264", str(video),
    ], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert 900 <= probe_duration_ms(video) <= 1100
    image, pts_ms = decode_frame(video, 400)
    assert image.size == (320, 240)
    assert abs(pts_ms - 400) <= 40

    rows = list(decode_frames(video, [0, 400, 800]))
    assert len(rows) == 3
    assert all(row.image.size == (320, 240) for row in rows)
    assert all(abs(row.pts_ms - row.requested_ms) <= 40 for row in rows)


@pytest.mark.gpu
def test_real_rapidocr_sessions_use_cuda():
    provider = RapidOCRProvider(Settings(execution_mode="cuda"))
    provider.ensure_ready()
    assert provider.status.actual_provider == CUDA_PROVIDER
    assert all(values[0] == CUDA_PROVIDER
               for values in provider.status.session_providers.values())
