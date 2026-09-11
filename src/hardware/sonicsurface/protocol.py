"""Pure SonicSurface phase quantization and UART frame encoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .profile import SonicSurfaceProfile


PHASE_BINS = 32
DISABLED_BIN = 32
FRAME_START = 254
FRAME_COMMIT = 253
ESP32_BOARD_PREFIX = 192


@dataclass(frozen=True)
class CompiledPattern:
    """Device-order phase bins and ready-to-write UART payload."""

    device_bins: np.ndarray
    frames: tuple[bytes, ...]

    @property
    def payload(self) -> bytes:
        return b"".join(self.frames)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def quantize_phases(phases_rad: np.ndarray) -> np.ndarray:
    """Round continuous radians to the nearest active SonicSurface phase bin."""
    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.ndim != 1 or not np.isfinite(phases).all():
        raise ValueError("phases_rad must be a finite one-dimensional vector")
    scaled = np.mod(phases, 2.0 * np.pi) * PHASE_BINS / (2.0 * np.pi)
    return np.floor(scaled + 0.5).astype(np.uint8) % PHASE_BINS


def encode_frames(protocol: str, device_bins: np.ndarray) -> tuple[bytes, ...]:
    """Encode documented vendor framing without opening a serial port."""
    bins = np.asarray(device_bins, dtype=np.uint8)
    if protocol == "fpga_256":
        if bins.shape != (256,):
            raise ValueError("fpga_256 requires exactly 256 device phase bins")
        return (bytes((FRAME_START,)) + bins.tobytes() + bytes((FRAME_COMMIT,)),)
    if protocol == "esp32_512":
        if bins.shape != (512,):
            raise ValueError("esp32_512 requires exactly 512 device phase bins")
        return (
            bytes((FRAME_START, ESP32_BOARD_PREFIX))
            + bins[:256].tobytes()
            + bytes((ESP32_BOARD_PREFIX + 1,))
            + bins[256:].tobytes()
            + bytes((FRAME_COMMIT,)),
        )
    raise ValueError(f"Unsupported SonicSurface protocol: {protocol}")


def compile_pattern(
    phases_rad: np.ndarray,
    profile: SonicSurfaceProfile,
) -> CompiledPattern:
    """Apply calibration, mapping and disabled channels to solver-order phases."""
    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.shape != (profile.solver_channels,):
        raise ValueError(
            f"Expected {profile.solver_channels} solver phases, got {phases.shape}"
        )
    if not np.isfinite(phases).all():
        raise ValueError("Solver phases must be finite")

    corrected = (
        profile.phase_sign * phases
        + profile.global_phase_offset_rad
        + profile.phase_offsets_rad
    )
    solver_bins = quantize_phases(corrected)
    device_bins = np.full(
        profile.device_channels,
        DISABLED_BIN,
        dtype=np.uint8,
    )
    device_bins[profile.solver_to_device] = solver_bins
    device_bins[profile.disabled_device_channels] = DISABLED_BIN
    return CompiledPattern(
        device_bins=device_bins,
        frames=encode_frames(profile.protocol, device_bins),
    )


def compile_all_off(profile: SonicSurfaceProfile) -> CompiledPattern:
    """Build the explicit all-disabled frame used for startup and emergency off."""
    device_bins = np.full(
        profile.device_channels,
        DISABLED_BIN,
        dtype=np.uint8,
    )
    return CompiledPattern(
        device_bins=device_bins,
        frames=encode_frames(profile.protocol, device_bins),
    )
