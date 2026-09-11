"""通过本地硬件服务发送 CoreEP4CE6 的诊断图样。"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hardware.sonicsurface.client import request_json


ARRAY_N = 16
DEFAULT_PITCH_M = 0.01
DEFAULT_FREQUENCY_HZ = 40_000.0
DEFAULT_SOUND_SPEED_M_S = 343.0


def diagnostic_phases(
    name: str,
    pitch_m: float,
    focus_z_m: float,
) -> np.ndarray:
    """按求解器的 row-major 阵元顺序生成 256 通道诊断相位。"""
    row, column = np.indices((ARRAY_N, ARRAY_N))
    if name == "uniform":
        phases = np.zeros((ARRAY_N, ARRAY_N))
    elif name == "checkerboard":
        phases = np.pi * ((row + column) % 2)
    elif name == "ramp_x":
        phases = 2.0 * np.pi * row / ARRAY_N
    elif name == "ramp_y":
        phases = 2.0 * np.pi * column / ARRAY_N
    elif name == "focus_center":
        coordinates = (np.arange(ARRAY_N) - (ARRAY_N - 1) / 2.0) * pitch_m
        x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
        distances = np.sqrt(x * x + y * y + focus_z_m * focus_z_m)
        wave_number = 2.0 * np.pi * DEFAULT_FREQUENCY_HZ / DEFAULT_SOUND_SPEED_M_S
        phases = -wave_number * distances
    else:
        raise ValueError(f"未知诊断图样: {name}")
    return np.asarray(phases, dtype=np.float64).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="经本地服务发送 16x16 CoreEP4CE6 诊断图样。"
    )
    parser.add_argument("--service-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--test",
        choices=(
            "off",
            "uniform",
            "checkerboard",
            "ramp_x",
            "ramp_y",
            "focus_center",
        ),
        required=True,
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="确认当前服务可实际发射；非关闭图样必须提供。",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="图样保持时间；结束后始终发送全关闭。",
    )
    parser.add_argument("--pitch-m", type=float, default=DEFAULT_PITCH_M)
    parser.add_argument("--focus-z-m", type=float, default=0.12)
    args = parser.parse_args()

    status = request_json(args.service_url, "GET", "/v1/status")
    if args.test == "off":
        print(request_json(args.service_url, "POST", "/v1/off", {}))
        return
    if not args.confirm_live:
        parser.error("非关闭图样必须显式提供 --confirm-live")
    if not status.get("live") or not status.get("armed"):
        parser.error("服务未处于已验证的 live 状态，拒绝发送诊断图样")
    if status.get("solver_channels") != ARRAY_N * ARRAY_N:
        parser.error("服务通道数不是 16x16=256，拒绝发送诊断图样")
    if args.hold_seconds <= 0:
        parser.error("--hold-seconds 必须为正数")

    phases = diagnostic_phases(args.test, args.pitch_m, args.focus_z_m)
    try:
        response = request_json(
            args.service_url,
            "POST",
            "/v1/pattern",
            {
                "phases_rad": phases.tolist(),
                "purpose": "diagnostic",
                "label": args.test,
            },
        )
        print(response)
        time.sleep(args.hold_seconds)
    finally:
        off_response = request_json(args.service_url, "POST", "/v1/off", {})
        print({"auto_off": off_response})


if __name__ == "__main__":
    main()
