"""Explicit serial transport. Importing this module never opens a port."""

from __future__ import annotations


class SerialTransport:
    """Lazy pyserial transport used only by an armed live service."""

    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self._connection = None

    def write(self, payload: bytes) -> None:
        if self._connection is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError(
                    "Live transmission requires pyserial. "
                    "Install it with: python -m pip install pyserial"
                ) from exc
            self._connection = serial.Serial(
                self.port,
                self.baudrate,
                write_timeout=2.0,
            )
        written = self._connection.write(payload)
        self._connection.flush()
        if written != len(payload):
            raise RuntimeError(
                f"Incomplete serial write: {written}/{len(payload)} bytes"
            )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
