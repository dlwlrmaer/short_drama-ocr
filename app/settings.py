from __future__ import annotations

import os
import platform
from dataclasses import dataclass


VALID_EXECUTION_MODES = {"auto", "cuda", "cpu"}
VALID_RUNTIME_PROFILES = {"auto", "win11", "linux-low-vram"}
VALID_MODEL_PROFILES = {"ppocrv6-small", "ppocrv5-mobile"}

PROFILE_DEFAULTS = {
    "win11": {"execution_mode": "auto", "model_profile": "ppocrv6-small"},
    "linux-low-vram": {"execution_mode": "cpu", "model_profile": "ppocrv5-mobile"},
}


def _number(name: str, default: str, cast):
    raw = os.getenv(name, default)
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 的值无效: {raw}") from exc


@dataclass(frozen=True)
class Settings:
    requested_profile: str = "win11"
    runtime_profile: str = "win11"
    model_profile: str = "ppocrv6-small"
    execution_mode: str = "auto"
    gpu_device_id: int = 0
    roi: tuple[float, float, float, float] = (0.10, 0.44, 0.85, 0.82)
    subtitle_center_band: tuple[float, float] = (0.715, 0.815)
    grid_ms: int = 1000
    dense_ms: int = 250
    padding_ms: int = 500
    trim_start_ms: int = 100
    trim_end_ms: int = 100
    decode_pts_guard_ms: int = 40
    min_inner_candidates: int = 3
    single_char_confirmation_gap_ms: int = 100

    def __post_init__(self) -> None:
        requested_profile = self.requested_profile.lower()
        runtime_profile = self.runtime_profile.lower()
        model_profile = self.model_profile.lower()
        mode = self.execution_mode.lower()
        object.__setattr__(self, "requested_profile", requested_profile)
        object.__setattr__(self, "runtime_profile", runtime_profile)
        object.__setattr__(self, "model_profile", model_profile)
        object.__setattr__(self, "execution_mode", mode)
        if requested_profile not in VALID_RUNTIME_PROFILES:
            raise ValueError(f"OCR_RUNTIME_PROFILE 必须是 {sorted(VALID_RUNTIME_PROFILES)}")
        if runtime_profile not in PROFILE_DEFAULTS:
            raise ValueError(f"实际 OCR profile 必须是 {sorted(PROFILE_DEFAULTS)}")
        if model_profile not in VALID_MODEL_PROFILES:
            raise ValueError(f"OCR_MODEL_PROFILE 必须是 {sorted(VALID_MODEL_PROFILES)}")
        if mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"OCR_EXECUTION_MODE 必须是 {sorted(VALID_EXECUTION_MODES)}")
        x0, y0, x1, y1 = self.roi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("字幕 ROI 必须是 0 到 1 之间的有效矩形")
        band0, band1 = self.subtitle_center_band
        if not (0 <= band0 < band1 <= 1):
            raise ValueError("字幕中心过滤带无效")
        positive = (self.grid_ms, self.dense_ms, self.min_inner_candidates,
                    self.single_char_confirmation_gap_ms)
        if any(value <= 0 for value in positive):
            raise ValueError("采样间隔、候选数和确认间隔必须大于 0")
        non_negative = (self.gpu_device_id, self.padding_ms, self.trim_start_ms,
                        self.trim_end_ms, self.decode_pts_guard_ms)
        if any(value < 0 for value in non_negative):
            raise ValueError("设备编号和时间参数不能为负数")

    @classmethod
    def for_profile(
        cls,
        profile_name: str = "auto",
        *,
        system_name: str | None = None,
        execution_mode: str | None = None,
        model_profile: str | None = None,
        gpu_device_id: int = 0,
    ) -> "Settings":
        requested = profile_name.lower()
        if requested not in VALID_RUNTIME_PROFILES:
            raise ValueError(f"OCR_RUNTIME_PROFILE 必须是 {sorted(VALID_RUNTIME_PROFILES)}")
        if requested == "auto":
            current_system = (system_name or platform.system()).lower()
            active = "win11" if current_system == "windows" else "linux-low-vram"
        else:
            active = requested
        defaults = PROFILE_DEFAULTS[active]
        return cls(
            requested_profile=requested,
            runtime_profile=active,
            model_profile=model_profile or defaults["model_profile"],
            execution_mode=execution_mode or defaults["execution_mode"],
            gpu_device_id=gpu_device_id,
        )

    @classmethod
    def from_env(cls, *, system_name: str | None = None) -> "Settings":
        return cls.for_profile(
            os.getenv("OCR_RUNTIME_PROFILE", "auto"),
            system_name=system_name,
            execution_mode=os.getenv("OCR_EXECUTION_MODE") or None,
            model_profile=os.getenv("OCR_MODEL_PROFILE") or None,
            gpu_device_id=_number("OCR_GPU_DEVICE_ID", "0", int),
        )
