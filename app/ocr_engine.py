from __future__ import annotations

import subprocess
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from .settings import Settings


MODEL_IDS = {
    "ppocrv6-small": "PP-OCRv6-small",
    "ppocrv5-mobile": "PP-OCRv5-mobile",
}
MODEL_ID = MODEL_IDS["ppocrv6-small"]
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class OCRBox:
    points: tuple[tuple[float, float], ...]
    text: str
    score: float


@dataclass
class ProviderStatus:
    ready: bool = False
    requested_profile: str = "auto"
    active_profile: str = "win11"
    requested_mode: str = "auto"
    actual_provider: str | None = None
    session_providers: dict[str, list[str]] | None = None
    gpu_device: str | None = None
    model: str = MODEL_ID
    rapidocr_version: str | None = None
    onnxruntime_version: str | None = None
    fallback_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_engine_factory(use_cuda: bool, device_id: int, model_profile: str):
    if use_cuda:
        import onnxruntime as ort

        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            # The GPU extra installs CUDA/cuDNN under site-packages on Windows.
            # Loading them before RapidOCR creates sessions avoids PATH mutation.
            preload(directory="")
    from rapidocr import ModelType, OCRVersion, RapidOCR

    model_variants = {
        "ppocrv6-small": (OCRVersion.PPOCRV6, ModelType.SMALL),
        "ppocrv5-mobile": (OCRVersion.PPOCRV5, ModelType.MOBILE),
    }
    ocr_version, model_type = model_variants[model_profile]

    return RapidOCR(params={
        "Det.ocr_version": ocr_version,
        "Det.model_type": model_type,
        "Rec.ocr_version": ocr_version,
        "Rec.model_type": model_type,
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": device_id,
    })


def _available_providers() -> list[str]:
    import onnxruntime as ort

    return ort.get_available_providers()


def _versions() -> tuple[str, str]:
    import importlib.metadata
    import onnxruntime as ort

    return importlib.metadata.version("rapidocr"), ort.__version__


def _gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def session_providers(engine: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name in ("text_det", "text_cls", "text_rec"):
        component = getattr(engine, name, None)
        wrapped = getattr(component, "session", None)
        session = getattr(wrapped, "session", wrapped)
        getter = getattr(session, "get_providers", None)
        if callable(getter):
            result[name] = list(getter())
    return result


def _all_cuda_first(providers: dict[str, list[str]]) -> bool:
    return bool(providers) and all(values and values[0] == CUDA_PROVIDER
                                   for values in providers.values())


class RapidOCRProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        engine_factory: Callable[[bool, int, str], Any] | None = None,
        available_provider_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._engine_factory = engine_factory or _default_engine_factory
        self._available_provider_fn = available_provider_fn or _available_providers
        self._engine: Any | None = None
        self._lock = threading.Lock()
        self.status = ProviderStatus(
            requested_profile=self.settings.requested_profile,
            active_profile=self.settings.runtime_profile,
            requested_mode=self.settings.execution_mode,
            model=MODEL_IDS[self.settings.model_profile],
        )

    def _make_and_check(self, use_cuda: bool) -> tuple[Any, dict[str, list[str]]]:
        engine = self._engine_factory(
            use_cuda, self.settings.gpu_device_id, self.settings.model_profile,
        )
        engine(np.full((64, 256, 3), 255, dtype=np.uint8))
        providers = session_providers(engine)
        if len(providers) != 3:
            raise RuntimeError("无法读取 RapidOCR 检测、分类和识别会话 Provider")
        if use_cuda and not _all_cuda_first(providers):
            raise RuntimeError(f"RapidOCR 实际会话未使用 CUDA: {providers}")
        return engine, providers

    def ensure_ready(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            mode = self.settings.execution_mode
            fallback_reason = None
            try:
                available = self._available_provider_fn()
                want_cuda = mode == "cuda" or (mode == "auto" and CUDA_PROVIDER in available)
                if mode == "cuda" and CUDA_PROVIDER not in available:
                    raise RuntimeError(f"CUDA Provider 不可用，当前 Provider: {available}")
                if mode == "auto" and CUDA_PROVIDER not in available:
                    fallback_reason = f"CUDA Provider 不可用，当前 Provider: {available}"
                if want_cuda:
                    try:
                        engine, providers = self._make_and_check(True)
                    except Exception as exc:
                        if mode == "cuda":
                            raise
                        fallback_reason = str(exc)
                        engine, providers = self._make_and_check(False)
                else:
                    engine, providers = self._make_and_check(False)
                rapid_version, ort_version = _versions()
                actual = CUDA_PROVIDER if _all_cuda_first(providers) else CPU_PROVIDER
                self._engine = engine
                self.status = ProviderStatus(
                    ready=True,
                    requested_profile=self.settings.requested_profile,
                    active_profile=self.settings.runtime_profile,
                    requested_mode=mode,
                    actual_provider=actual,
                    session_providers=providers, gpu_device=_gpu_name() if actual == CUDA_PROVIDER else None,
                    model=MODEL_IDS[self.settings.model_profile],
                    rapidocr_version=rapid_version, onnxruntime_version=ort_version,
                    fallback_reason=fallback_reason,
                )
                return engine
            except Exception as exc:
                self.status.ready = False
                self.status.error = str(exc)
                raise

    def recognize(self, image: Any) -> list[OCRBox]:
        engine = self.ensure_ready()
        output = engine(image)
        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if boxes is None or texts is None or scores is None:
            return []
        rows = []
        for points, text, score in zip(boxes, texts, scores):
            normalized_points = tuple((float(point[0]), float(point[1])) for point in points)
            rows.append(OCRBox(normalized_points, str(text), float(score)))
        return rows

    def text(self, image: Any) -> str:
        rows = self.recognize(image)
        return "".join(row.text for row in rows).strip()


_provider: RapidOCRProvider | None = None
_provider_lock = threading.Lock()


def get_provider() -> RapidOCRProvider:
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = RapidOCRProvider()
    return _provider


def reset_provider() -> None:
    global _provider
    with _provider_lock:
        _provider = None
