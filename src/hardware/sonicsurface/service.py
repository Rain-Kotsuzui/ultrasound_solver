"""Loopback-only HTTP service that owns SonicSurface serial access."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .profile import SonicSurfaceProfile, load_profile
from .protocol import CompiledPattern, compile_all_off, compile_pattern
from .service_logging import configure_service_logging
from .transport import SerialTransport


LOGGER = logging.getLogger("hardware.sonicsurface.service")


class SonicSurfaceController:
    """Serial-owner state machine. It starts disabled and is live only if armed."""

    def __init__(
        self,
        profile: SonicSurfaceProfile,
        live: bool,
        off_only: bool = False,
        calibration_mode: bool = False,
    ):
        self.profile = profile
        self.live = live
        self.off_only = off_only
        self.calibration_mode = calibration_mode
        self.last_pattern: CompiledPattern | None = None
        if (
            live
            and not off_only
            and not calibration_mode
            and not profile.mapping_verified
        ):
            raise ValueError(
                "Live service requires mapping_verified: true in the profile"
            )
        self.transport = (
            SerialTransport(profile.serial_port, profile.baudrate)
            if live and profile.serial_port
            else None
        )
        if live and self.transport is None:
            raise ValueError("Live service requires serial.port in the profile")

    def status(self) -> dict[str, Any]:
        return {
            "service": "sonicsurface",
            "profile": self.profile.name,
            "board_model": self.profile.board_model,
            "protocol": self.profile.protocol,
            "solver_channels": self.profile.solver_channels,
            "device_channels": self.profile.device_channels,
            "mapping_verified": self.profile.mapping_verified,
            "live": self.live,
            "armed": self.live and not self.off_only,
            "off_only": self.off_only,
            "calibration_mode": self.calibration_mode,
            "last_pattern_sha256": (
                self.last_pattern.sha256 if self.last_pattern else None
            ),
        }

    def submit(
        self,
        phases: np.ndarray,
        source: str = "unknown",
        purpose: str = "optimization",
        label: str | None = None,
    ) -> dict[str, Any]:
        if self.off_only:
            raise PermissionError(
                "This live service is off-only until the profile mapping is verified"
            )
        if self.calibration_mode and purpose != "diagnostic":
            raise PermissionError(
                "Calibration mode accepts only diagnostic patterns"
            )
        pattern = compile_pattern(phases, self.profile)
        if self.live:
            self.transport.write(pattern.payload)
        self.last_pattern = pattern
        response = self._pattern_response(pattern)
        LOGGER.info(
            "相位图样已接受",
            extra={
                "event": "pattern_accepted",
                "details": {
                    **response,
                    "source": source,
                    "solver_channels": int(phases.size),
                    "purpose": purpose,
                    "label": label,
                    "board_model": self.profile.board_model,
                    "profile": self.profile.name,
                },
            },
        )
        return response

    def off(self, source: str = "unknown") -> dict[str, Any]:
        pattern = compile_all_off(self.profile)
        if self.live:
            self.transport.write(pattern.payload)
        self.last_pattern = pattern
        response = self._pattern_response(pattern)
        LOGGER.warning(
            "已请求全关闭图样",
            extra={
                "event": "array_off",
                "details": {
                    **response,
                    "source": source,
                    "board_model": self.profile.board_model,
                    "profile": self.profile.name,
                },
            },
        )
        return response

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()

    def _pattern_response(self, pattern: CompiledPattern) -> dict[str, Any]:
        return {
            "accepted": True,
            "transmitted": self.live,
            "protocol": self.profile.protocol,
            "frame_bytes": len(pattern.payload),
            "payload_sha256": pattern.sha256,
            "disabled_channels": int(
                np.count_nonzero(pattern.device_bins == 32)
            ),
        }


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(controller: SonicSurfaceController):
    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:
            if self.path == "/v1/status":
                _json_response(self, HTTPStatus.OK, controller.status())
                return
            _json_response(
                self,
                HTTPStatus.NOT_FOUND,
                {"error": "Unknown endpoint"},
            )

        def do_POST(self) -> None:
            try:
                source = self.client_address[0]
                if self.path == "/v1/off":
                    _json_response(
                        self,
                        HTTPStatus.OK,
                        controller.off(source),
                    )
                    return
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1_000_000:
                    raise ValueError("Invalid request body length")
                raw = self.rfile.read(content_length)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                if self.path == "/v1/pattern":
                    phases = np.asarray(payload["phases_rad"], dtype=np.float64)
                    purpose = str(payload.get("purpose", "optimization"))
                    label = payload.get("label")
                    if label is not None:
                        label = str(label)
                    _json_response(
                        self,
                        HTTPStatus.OK,
                        controller.submit(phases, source, purpose, label),
                    )
                    return
                _json_response(
                    self,
                    HTTPStatus.NOT_FOUND,
                    {"error": "Unknown endpoint"},
                )
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning(
                    "请求参数无效",
                    extra={
                        "event": "request_rejected",
                        "details": {
                            "path": self.path,
                            "source": self.client_address[0],
                            "reason": str(exc),
                            "board_model": controller.profile.board_model,
                            "profile": controller.profile.name,
                        },
                    },
                )
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(exc)},
                )
            except PermissionError as exc:
                LOGGER.warning(
                    "请求被安全策略拒绝",
                    extra={
                        "event": "request_forbidden",
                        "details": {
                            "path": self.path,
                            "source": self.client_address[0],
                            "reason": str(exc),
                            "board_model": controller.profile.board_model,
                            "profile": controller.profile.name,
                        },
                    },
                )
                _json_response(
                    self,
                    HTTPStatus.FORBIDDEN,
                    {"error": str(exc)},
                )
            except Exception as exc:
                LOGGER.exception(
                    "图样处理失败",
                    extra={
                        "event": "request_failed",
                        "details": {
                            "path": self.path,
                            "source": self.client_address[0],
                            "reason": str(exc),
                            "board_model": controller.profile.board_model,
                            "profile": controller.profile.name,
                        },
                    },
                )
                _json_response(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Pattern rejected: {exc}"},
                )

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return RequestHandler


def serve(
    profile_path: str | Path,
    port: int,
    live: bool,
    off_only: bool = False,
    calibration_mode: bool = False,
    log_file: str | Path = "outputs/hardware/sonicsurface_service.log",
    log_level: str = "INFO",
) -> None:
    configure_service_logging(log_file, log_level)
    profile = load_profile(profile_path)
    controller = SonicSurfaceController(
        profile,
        live=live,
        off_only=off_only,
        calibration_mode=calibration_mode,
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        make_handler(controller),
    )
    LOGGER.info(
        "服务已启动",
        extra={
            "event": "service_started",
            "details": {
                "host": "127.0.0.1",
                "port": port,
                "live": live,
                "off_only": off_only,
                "calibration_mode": calibration_mode,
                "profile": profile.name,
                "board_model": profile.board_model,
                "protocol": profile.protocol,
                "log_file": str(log_file),
            },
        },
    )
    try:
        server.serve_forever()
    finally:
        LOGGER.info(
            "服务已停止",
            extra={
                "event": "service_stopped",
                "details": {
                    "profile": profile.name,
                    "board_model": profile.board_model,
                },
            },
        )
        controller.close()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the loopback-only SonicSurface hardware service."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--log-file",
        default="outputs/hardware/sonicsurface_service.log",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Open the configured serial port and transmit accepted frames.",
    )
    parser.add_argument(
        "--live-off-only",
        action="store_true",
        help=(
            "Open the serial port but accept only /v1/off. This is intended "
            "for a safe USB/UART connection check before mapping verification."
        ),
    )
    parser.add_argument(
        "--live-calibration",
        action="store_true",
        help=(
            "Allow only short diagnostic patterns with an unverified profile. "
            "Do not use this mode to transmit optimized phases."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if (args.live_off_only or args.live_calibration) and not args.live:
        parser.error("--live-off-only/--live-calibration requires --live")
    if args.live_off_only and args.live_calibration:
        parser.error("--live-off-only and --live-calibration cannot be combined")
    serve(
        args.profile,
        args.port,
        args.live,
        args.live_off_only,
        args.live_calibration,
        args.log_file,
        args.log_level,
    )


if __name__ == "__main__":
    main()
