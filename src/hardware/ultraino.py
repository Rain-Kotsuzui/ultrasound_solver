"""Ultraino SimpleFPGA adapter and loopback-only phase service.

The wire protocol mirrors third_party/Ultraino/.../protocols/SimpleFPGA.java.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CHANNELS = 256
PHASE_DIVISIONS = 32
PHASE_OFF = 32
START_PHASES = 254
SWAP_BUFFERS = 253
DEFAULT_BAUDRATE = 230400


@dataclass(frozen=True)
class UltrainoConfig:
    name: str
    serial_port: str | None
    baudrate: int
    solver_to_order: np.ndarray
    phase_corrections_pi: np.ndarray
    calibration_verified: bool


def _permutation(raw: object) -> np.ndarray:
    if raw == "identity":
        return np.arange(CHANNELS, dtype=np.int64)
    values = np.asarray(raw, dtype=np.int64)
    if values.shape != (CHANNELS,) or set(values.tolist()) != set(range(CHANNELS)):
        raise ValueError("solver_to_ultraino_order 必须是 0..255 的完整排列")
    return values


def load_config(path: str | Path) -> UltrainoConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("不支持的 Ultraino 硬件配置")
    serial = raw.get("serial", {})
    if not isinstance(serial, dict):
        raise ValueError("serial 必须是 YAML 映射")
    raw_corrections = raw.get("phase_corrections_pi", "zeros")
    corrections = (
        np.zeros(CHANNELS, dtype=np.float64)
        if raw_corrections == "zeros"
        else np.asarray(raw_corrections, dtype=np.float64)
    )
    if corrections.shape != (CHANNELS,) or not np.isfinite(corrections).all():
        raise ValueError("phase_corrections_pi 必须包含 256 个有限数值")
    baudrate = int(serial.get("baudrate", DEFAULT_BAUDRATE))
    if baudrate <= 0:
        raise ValueError("serial.baudrate 必须为正数")
    port = serial.get("port")
    return UltrainoConfig(
        name=str(raw.get("name", Path(path).stem)),
        serial_port=None if port is None else str(port),
        baudrate=baudrate,
        solver_to_order=_permutation(raw.get("solver_to_ultraino_order", "identity")),
        phase_corrections_pi=corrections,
        calibration_verified=bool(raw.get("calibration_verified", False)),
    )


def _java_round(values: np.ndarray) -> np.ndarray:
    """Match Java Math.round for the finite values used by SimpleFPGA."""
    return np.floor(values + 0.5).astype(np.int64)


def compile_simple_fpga(
    phases_rad: np.ndarray,
    config: UltrainoConfig,
) -> tuple[bytes, bytes]:
    """Build the two writes performed by Ultraino SimpleFPGA.sendPattern()."""
    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.shape != (CHANNELS,) or not np.isfinite(phases).all():
        raise ValueError("需要 256 个有限的连续相位（rad）")
    normalized_pi = phases / np.pi + config.phase_corrections_pi
    bins = _java_round(normalized_pi * PHASE_DIVISIONS / 2.0) % PHASE_DIVISIONS
    device_bins = np.full(CHANNELS, PHASE_OFF, dtype=np.uint8)
    device_bins[config.solver_to_order] = bins.astype(np.uint8)
    # Ultraino writes 0xFE + 256 data bytes, then a separate 0xFD buffer swap.
    return bytes((START_PHASES,)) + device_bins.tobytes(), bytes((SWAP_BUFFERS,))


def compile_all_off() -> tuple[bytes, bytes]:
    return (
        bytes((START_PHASES,)) + bytes((PHASE_OFF,)) * CHANNELS,
        bytes((SWAP_BUFFERS,)),
    )


class UltrainoController:
    def __init__(self, config: UltrainoConfig, live: bool):
        if live and not config.calibration_verified:
            raise ValueError(
                "真实发送要求 calibration_verified: true；"
                "请先在 AcousticField 完成通道和相位标定"
            )
        if live and not config.serial_port:
            raise ValueError("真实发送要求 serial.port")
        self.config = config
        self.live = live
        self.serial = None
        if live:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("缺少 pyserial，无法打开串口") from exc
            self.serial = serial.Serial(
                config.serial_port,
                config.baudrate,
                timeout=1,
                write_timeout=1,
            )

    def status(self) -> dict[str, Any]:
        return {
            "service": "ultraino-simple-fpga",
            "config": self.config.name,
            "channels": CHANNELS,
            "phase_divisions": PHASE_DIVISIONS,
            "baudrate": self.config.baudrate,
            "calibration_verified": self.config.calibration_verified,
            "live": self.live,
            "protocol_source": (
                "third_party/Ultraino/AcousticFieldSim/src/"
                "acousticfield3d/protocols/SimpleFPGA.java"
            ),
        }

    def _send(self, phase_write: bytes, swap_write: bytes) -> bool:
        if not self.live:
            return False
        assert self.serial is not None
        self.serial.write(phase_write)
        self.serial.write(swap_write)
        self.serial.flush()
        return True

    def submit(self, phases: np.ndarray) -> dict[str, Any]:
        phase_write, swap_write = compile_simple_fpga(phases, self.config)
        transmitted = self._send(phase_write, swap_write)
        return {
            "accepted": True,
            "transmitted": transmitted,
            "phase_write_bytes": len(phase_write),
            "swap_write_bytes": len(swap_write),
            "payload_sha256": hashlib.sha256(phase_write + swap_write).hexdigest(),
        }

    def off(self) -> dict[str, Any]:
        phase_write, swap_write = compile_all_off()
        transmitted = self._send(phase_write, swap_write)
        return {
            "accepted": True,
            "transmitted": transmitted,
            "phase_write_bytes": len(phase_write),
            "swap_write_bytes": len(swap_write),
            "payload_sha256": hashlib.sha256(phase_write + swap_write).hexdigest(),
        }

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()


def _respond(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def serve(config: UltrainoConfig, port: int, live: bool) -> None:
    controller = UltrainoController(config, live)
    logger = logging.getLogger("hardware.ultraino")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/status":
                _respond(self, HTTPStatus.OK, controller.status())
            else:
                _respond(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length))
                if self.path == "/v1/pattern":
                    result = controller.submit(np.asarray(payload["phases_rad"]))
                    logger.info("pattern accepted %s", result["payload_sha256"])
                elif self.path == "/v1/off":
                    result = controller.off()
                    logger.warning("all-off requested")
                else:
                    _respond(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                _respond(self, HTTPStatus.OK, result)
            except (KeyError, TypeError, ValueError) as exc:
                _respond(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    logger.info("Ultraino service started on 127.0.0.1:%d live=%s", port, live)
    try:
        server.serve_forever()
    finally:
        controller.close()
        server.server_close()


def service_main() -> None:
    parser = argparse.ArgumentParser(description="运行 Ultraino SimpleFPGA 本地服务。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    serve(load_config(args.config), args.port, args.live)
