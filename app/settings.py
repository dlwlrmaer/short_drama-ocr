from __future__ import annotations

import os
import platform
import subprocess
import ctypes
from dataclasses import dataclass


VALID_EXECUTION_MODES = {"auto", "cuda", "cpu"}
VALID_RUNTIME_PROFILES = {"auto", "win11", "linux", "linux-low-vram"}
VALID_MODEL_PROFILES = {"ppocrv6-small", "ppocrv5-mobile"}
VALID_BACKGROUND_SUPPRESSION = {"off", "spatial", "adaptive"}

PROFILE_DEFAULTS = {
    "win11": {
        "execution_mode": "auto", "model_profile": "ppocrv6-small",
        "background_suppression": "adaptive",
    },
    "linux": {
        "execution_mode": "auto", "model_profile": "ppocrv6-small",
        "background_suppression": "adaptive",
    },
    "linux-low-vram": {
        "execution_mode": "cpu", "model_profile": "ppocrv5-mobile",
        "background_suppression": "spatial",
    },
}


@dataclass(frozen=True)
class HardwareInfo:
    system_name: str
    total_memory_mb: int | None = None
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None


def _total_memory_mb(system_name: str) -> int | None:
    if system_name == "windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong), ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.total_phys / 1024 / 1024)
        except (AttributeError, OSError):
            return None
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 / 1024)
    except (AttributeError, OSError, ValueError):
        return None


def detect_hardware(system_name: str | None = None) -> HardwareInfo:
    system = (system_name or platform.system()).lower()
    gpu_name = None
    gpu_memory_mb = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, memory = [part.strip() for part in result.stdout.splitlines()[0].rsplit(",", 1)]
            gpu_name, gpu_memory_mb = name, int(memory)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return HardwareInfo(system, _total_memory_mb(system), gpu_name, gpu_memory_mb)


def _batch_limit(memory_mb: int | None, active_profile: str) -> int:
    if memory_mb is not None and memory_mb < 8192:
        return 8
    if memory_mb is not None and memory_mb < 16384:
        return 16
    return 16 if active_profile == "linux-low-vram" else 32


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
    hardware_tier: str = "standard"
    total_memory_mb: int | None = None
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    max_batch_images: int = 32
    background_suppression: str = "off"
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
        suppression = self.background_suppression.lower()
        mode = self.execution_mode.lower()
        object.__setattr__(self, "requested_profile", requested_profile)
        object.__setattr__(self, "runtime_profile", runtime_profile)
        object.__setattr__(self, "model_profile", model_profile)
        object.__setattr__(self, "background_suppression", suppression)
        object.__setattr__(self, "execution_mode", mode)
        if requested_profile not in VALID_RUNTIME_PROFILES:
            raise ValueError(f"OCR_RUNTIME_PROFILE 必须是 {sorted(VALID_RUNTIME_PROFILES)}")
        if runtime_profile not in PROFILE_DEFAULTS:
            raise ValueError(f"实际 OCR profile 必须是 {sorted(PROFILE_DEFAULTS)}")
        if model_profile not in VALID_MODEL_PROFILES:
            raise ValueError(f"OCR_MODEL_PROFILE 必须是 {sorted(VALID_MODEL_PROFILES)}")
        if suppression not in VALID_BACKGROUND_SUPPRESSION:
            raise ValueError(
                f"OCR_BACKGROUND_SUPPRESSION 必须是 {sorted(VALID_BACKGROUND_SUPPRESSION)}"
            )
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
        if not 1 <= self.max_batch_images <= 64:
            raise ValueError("OCR_MAX_BATCH_IMAGES 必须在 1 到 64 之间")

    @classmethod
    def for_profile(
        cls,
        profile_name: str = "auto",
        *,
        system_name: str | None = None,
        execution_mode: str | None = None,
        model_profile: str | None = None,
        gpu_device_id: int = 0,
        background_suppression: str | None = None,
        max_batch_images: int | None = None,
        hardware: HardwareInfo | None = None,
    ) -> "Settings":
        requested = profile_name.lower()
        if requested not in VALID_RUNTIME_PROFILES:
            raise ValueError(f"OCR_RUNTIME_PROFILE 必须是 {sorted(VALID_RUNTIME_PROFILES)}")
        detected = hardware or detect_hardware(system_name)
        if requested == "auto":
            active = "win11" if detected.system_name == "windows" else "linux"
        else:
            active = requested
        if active == "linux" and (detected.gpu_memory_mb or 0) < 4096:
            active = "linux-low-vram"
        defaults = PROFILE_DEFAULTS[active]
        low_resource = active == "linux-low-vram" or (
            detected.total_memory_mb is not None and detected.total_memory_mb < 8192
        )
        return cls(
            requested_profile=requested,
            runtime_profile=active,
            model_profile=model_profile or defaults["model_profile"],
            execution_mode=execution_mode or defaults["execution_mode"],
            gpu_device_id=gpu_device_id,
            hardware_tier="low-resource" if low_resource else "accelerated",
            total_memory_mb=detected.total_memory_mb,
            gpu_name=detected.gpu_name,
            gpu_memory_mb=detected.gpu_memory_mb,
            max_batch_images=(
                max_batch_images
                if max_batch_images is not None
                else _batch_limit(detected.total_memory_mb, active)
            ),
            background_suppression=(
                background_suppression or defaults["background_suppression"]
            ),
        )

    @classmethod
    def from_env(
        cls, *, system_name: str | None = None, hardware: HardwareInfo | None = None,
    ) -> "Settings":
        return cls.for_profile(
            os.getenv("OCR_RUNTIME_PROFILE", "auto"),
            system_name=system_name,
            execution_mode=os.getenv("OCR_EXECUTION_MODE") or None,
            model_profile=os.getenv("OCR_MODEL_PROFILE") or None,
            gpu_device_id=_number("OCR_GPU_DEVICE_ID", "0", int),
            background_suppression=os.getenv("OCR_BACKGROUND_SUPPRESSION") or None,
            max_batch_images=(
                _number("OCR_MAX_BATCH_IMAGES", "32", int)
                if os.getenv("OCR_MAX_BATCH_IMAGES") else None
            ),
            hardware=hardware,
        )
