from types import SimpleNamespace

import numpy as np
import pytest

import app.ocr_engine as engine_module
from app.ocr_engine import CPU_PROVIDER, CUDA_PROVIDER, RapidOCRProvider, session_providers
from app.settings import Settings


class FakeOrtSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class FakeEngine:
    def __init__(self, providers, result=None):
        for name in ("text_det", "text_cls", "text_rec"):
            setattr(self, name, SimpleNamespace(
                session=SimpleNamespace(session=FakeOrtSession(providers))))
        self.result = result or SimpleNamespace(boxes=None, txts=None, scores=None)

    def __call__(self, _image):
        return self.result


def test_auto_prefers_cuda_and_reuses_singleton(monkeypatch):
    calls = []

    def factory(use_cuda, device, model_profile):
        calls.append((use_cuda, device, model_profile))
        return FakeEngine([CUDA_PROVIDER, CPU_PROVIDER] if use_cuda else [CPU_PROVIDER])

    monkeypatch.setattr(engine_module, "_versions", lambda: ("3.9.2", "1.30.0"))
    monkeypatch.setattr(engine_module, "_gpu_name", lambda: "RTX 4070")
    provider = RapidOCRProvider(Settings(execution_mode="auto"), factory,
                                lambda: [CUDA_PROVIDER, CPU_PROVIDER])
    assert provider.ensure_ready() is provider.ensure_ready()
    assert calls == [(True, 0, "ppocrv6-small")]
    assert provider.status.actual_provider == CUDA_PROVIDER
    assert provider.status.gpu_device == "RTX 4070"


def test_auto_falls_back_but_explicit_cuda_fails(monkeypatch):
    monkeypatch.setattr(engine_module, "_versions", lambda: ("3.9.2", "1.30.0"))

    def factory(use_cuda, _device, _model_profile):
        return FakeEngine([CPU_PROVIDER])

    auto = RapidOCRProvider(Settings(execution_mode="auto"), factory,
                            lambda: [CUDA_PROVIDER, CPU_PROVIDER])
    auto.ensure_ready()
    assert auto.status.actual_provider == CPU_PROVIDER
    assert "未使用 CUDA" in auto.status.fallback_reason

    explicit = RapidOCRProvider(Settings(execution_mode="cuda"), factory,
                                lambda: [CUDA_PROVIDER, CPU_PROVIDER])
    with pytest.raises(RuntimeError, match="未使用 CUDA"):
        explicit.ensure_ready()
    assert explicit.status.ready is False


def test_cpu_mode_disables_cuda_and_converts_output(monkeypatch):
    calls = []
    result = SimpleNamespace(
        boxes=[[[1, 2], [3, 2], [3, 4], [1, 4]]], txts=["妈"], scores=[0.98])

    def factory(use_cuda, _device, model_profile):
        calls.append((use_cuda, model_profile))
        return FakeEngine([CPU_PROVIDER], result)

    monkeypatch.setattr(engine_module, "_versions", lambda: ("3.9.2", "1.30.0"))
    provider = RapidOCRProvider(Settings(execution_mode="cpu"), factory,
                                lambda: [CUDA_PROVIDER, CPU_PROVIDER])
    boxes = provider.recognize(np.zeros((8, 8, 3), dtype=np.uint8))
    assert calls == [(False, "ppocrv6-small")]
    assert boxes[0].text == "妈" and boxes[0].score == pytest.approx(0.98)
    assert session_providers(provider.ensure_ready())["text_det"] == [CPU_PROVIDER]


def test_linux_low_vram_profile_uses_mobile_model_and_reports_profile(monkeypatch):
    calls = []

    def factory(use_cuda, _device, model_profile):
        calls.append((use_cuda, model_profile))
        return FakeEngine([CPU_PROVIDER])

    monkeypatch.setattr(engine_module, "_versions", lambda: ("3.9.2", "1.30.0"))
    settings = Settings.for_profile("linux-low-vram")
    provider = RapidOCRProvider(settings, factory, lambda: [CPU_PROVIDER])
    provider.ensure_ready()

    assert calls == [(False, "ppocrv5-mobile")]
    assert provider.status.requested_profile == "linux-low-vram"
    assert provider.status.active_profile == "linux-low-vram"
    assert provider.status.model == "PP-OCRv5-mobile"
