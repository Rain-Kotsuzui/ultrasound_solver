"""Validated SonicSurface deployment profiles."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


PROTOCOL_CHANNELS = {
    "fpga_256": 256,
    "esp32_512": 512,
}
SUPPORTED_BOARD_MODELS = {"coreep4ce6"}
VENDOR_SONICSURFACE_16X16 = "vendor_sonicsurface_16x16"


@dataclass(frozen=True)
class SonicSurfaceProfile:
    """Hardware mapping and calibration required to encode one phase vector."""

    name: str
    board_model: str
    protocol: str
    solver_channels: int
    solver_to_device: np.ndarray
    phase_offsets_rad: np.ndarray
    phase_sign: int
    global_phase_offset_rad: float
    disabled_device_channels: np.ndarray
    mapping_verified: bool
    serial_port: str | None
    baudrate: int

    @property
    def device_channels(self) -> int:
        return PROTOCOL_CHANNELS[self.protocol]

    @property
    def enabled_device_channels(self) -> int:
        return self.device_channels - int(self.disabled_device_channels.size)


def _int_vector(raw, name: str) -> np.ndarray:
    values = np.asarray(raw, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional list")
    return values


def _validate_permutation(mapping: np.ndarray, channels: int) -> None:
    if mapping.shape != (channels,):
        raise ValueError(
            "solver_to_device must have one entry for every solver channel"
        )
    if np.any(mapping < 0) or np.any(mapping >= channels):
        raise ValueError("solver_to_device contains an out-of-range channel")
    if np.unique(mapping).size != channels:
        raise ValueError("solver_to_device must be a bijective permutation")


def _vendor_sonicsurface_mapping() -> np.ndarray:
    """读取供应商 Python API 中的 EMITTERS_ORDER，避免复制 256 项排列。"""
    source_path = (
        Path(__file__).resolve().parents[3]
        / "third_party"
        / "SonicSurface"
        / "ControlSoftware"
        / "Python"
        / "SonicSurface.py"
    )
    if not source_path.exists():
        raise FileNotFoundError(
            "未找到供应商 EMITTERS_ORDER："
            f"{source_path}"
        )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "EMITTERS_ORDER"
                for target in node.targets
            )
        ):
            mapping = _int_vector(
                ast.literal_eval(node.value),
                "供应商 EMITTERS_ORDER",
            )
            _validate_permutation(mapping, 256)
            return mapping
    raise ValueError("供应商 SonicSurface.py 中未找到 EMITTERS_ORDER")


def load_profile(path: str | Path) -> SonicSurfaceProfile:
    """Load a strict profile; partial mappings and implicit padding are rejected."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Hardware profile must be a YAML mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported hardware profile schema_version")

    board_model = str(raw.get("board_model", "")).lower()
    if board_model not in SUPPORTED_BOARD_MODELS:
        raise ValueError(
            "board_model must be coreep4ce6 for the supported SonicSurface "
            "hardware profile"
        )

    protocol = str(raw.get("protocol", "")).lower()
    if protocol not in PROTOCOL_CHANNELS:
        raise ValueError(
            f"protocol must be one of {', '.join(PROTOCOL_CHANNELS)}"
        )
    channels = PROTOCOL_CHANNELS[protocol]
    solver_channels = int(raw.get("solver_channels", 0))
    if solver_channels != channels:
        raise ValueError(
            f"{protocol} requires {channels} solver channels; got "
            f"{solver_channels}. Do not pad or repeat optimized phases."
        )

    raw_mapping = raw.get("solver_to_device", [])
    if raw_mapping == "identity":
        mapping = np.arange(channels, dtype=np.int64)
    elif raw_mapping == VENDOR_SONICSURFACE_16X16:
        if protocol != "fpga_256":
            raise ValueError(
                "vendor_sonicsurface_16x16 仅适用于 fpga_256"
            )
        mapping = _vendor_sonicsurface_mapping()
    else:
        mapping = _int_vector(raw_mapping, "solver_to_device")
    _validate_permutation(mapping, channels)

    offsets = np.asarray(
        raw.get("phase_offsets_rad", np.zeros(channels)),
        dtype=np.float64,
    )
    if offsets.shape != (channels,) or not np.isfinite(offsets).all():
        raise ValueError(
            "phase_offsets_rad must be finite with one value per solver channel"
        )

    phase_sign = int(raw.get("phase_sign", 1))
    if phase_sign not in {-1, 1}:
        raise ValueError("phase_sign must be either 1 or -1")
    global_offset = float(raw.get("global_phase_offset_rad", 0.0))
    if not np.isfinite(global_offset):
        raise ValueError("global_phase_offset_rad must be finite")

    disabled = _int_vector(
        raw.get("disabled_device_channels", []),
        "disabled_device_channels",
    )
    if np.any(disabled < 0) or np.any(disabled >= channels):
        raise ValueError("disabled_device_channels contains an out-of-range channel")
    if np.unique(disabled).size != disabled.size:
        raise ValueError("disabled_device_channels contains duplicates")

    serial = raw.get("serial", {})
    if not isinstance(serial, dict):
        raise ValueError("serial must be a mapping")
    serial_port = serial.get("port")
    if serial_port is not None:
        serial_port = str(serial_port)
    baudrate = int(serial.get("baudrate", 230400))
    if baudrate <= 0:
        raise ValueError("serial.baudrate must be positive")

    return SonicSurfaceProfile(
        name=str(raw.get("name", path.stem)),
        board_model=board_model,
        protocol=protocol,
        solver_channels=solver_channels,
        solver_to_device=mapping,
        phase_offsets_rad=offsets,
        phase_sign=phase_sign,
        global_phase_offset_rad=global_offset,
        disabled_device_channels=disabled,
        mapping_verified=bool(raw.get("mapping_verified", False)),
        serial_port=serial_port,
        baudrate=baudrate,
    )
