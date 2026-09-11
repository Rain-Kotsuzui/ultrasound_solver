"""列出 Windows 中可用于 SonicSurface 的 USB 串口。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="列出 USB/UART 串口。")
    parser.parse_args()
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise SystemExit(
            "缺少 pyserial。请执行: python -m pip install pyserial"
        ) from exc

    ports = list(list_ports.comports())
    if not ports:
        print("未发现串口。请检查 USB 数据线、驱动和控制板供电。")
        return
    for port in ports:
        details = " | ".join(
            item for item in (port.description, port.hwid) if item
        )
        print(f"{port.device}: {details}")


if __name__ == "__main__":
    main()
