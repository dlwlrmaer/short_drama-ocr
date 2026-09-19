import pytest

from app.settings import Settings


def test_execution_modes_are_normalized_and_validated(monkeypatch):
    monkeypatch.setenv("OCR_RUNTIME_PROFILE", "win11")
    monkeypatch.setenv("OCR_EXECUTION_MODE", "CUDA")
    monkeypatch.setenv("OCR_GPU_DEVICE_ID", "1")
    settings = Settings.from_env()
    assert settings.execution_mode == "cuda"
    assert settings.gpu_device_id == 1
    with pytest.raises(ValueError):
        Settings(execution_mode="directml")


def test_auto_profile_selects_host_defaults(monkeypatch):
    monkeypatch.delenv("OCR_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("OCR_MODEL_PROFILE", raising=False)

    windows = Settings.from_env(system_name="Windows")
    assert windows.requested_profile == "auto"
    assert windows.runtime_profile == "win11"
    assert windows.model_profile == "ppocrv6-small"
    assert windows.execution_mode == "auto"

    linux = Settings.from_env(system_name="Linux")
    assert linux.requested_profile == "auto"
    assert linux.runtime_profile == "linux-low-vram"
    assert linux.model_profile == "ppocrv5-mobile"
    assert linux.execution_mode == "cpu"


def test_profile_defaults_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OCR_RUNTIME_PROFILE", "linux-low-vram")
    monkeypatch.setenv("OCR_EXECUTION_MODE", "cuda")
    monkeypatch.setenv("OCR_MODEL_PROFILE", "ppocrv6-small")

    settings = Settings.from_env(system_name="Linux")
    assert settings.runtime_profile == "linux-low-vram"
    assert settings.execution_mode == "cuda"
    assert settings.model_profile == "ppocrv6-small"

    with pytest.raises(ValueError):
        Settings.for_profile("nas-large")
    with pytest.raises(ValueError):
        Settings(model_profile="unknown")


def test_frozen_defaults_and_parameter_boundaries():
    settings = Settings()
    assert settings.roi == (0.10, 0.44, 0.85, 0.82)
    assert settings.subtitle_center_band == (0.715, 0.815)
    assert (settings.grid_ms, settings.dense_ms, settings.padding_ms) == (1000, 250, 500)
    assert (settings.trim_start_ms, settings.trim_end_ms) == (100, 100)
    assert settings.decode_pts_guard_ms == 40
    with pytest.raises(ValueError):
        Settings(roi=(0.8, 0.4, 0.2, 0.9))
    with pytest.raises(ValueError):
        Settings(dense_ms=0)
