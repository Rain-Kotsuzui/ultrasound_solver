"""将 AcousticField 导出的 32 档相位修正写入 Ultraino 服务配置。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import yaml


CHANNELS = 256
DIVISIONS = 32


def read_integer_vector(path: Path, label: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8-sig")
    values = np.asarray(
        [int(value) for value in re.findall(r"-?\d+", text)],
        dtype=np.int64,
    )
    if values.shape != (CHANNELS,):
        raise ValueError(f"{label} 必须恰好包含 {CHANNELS} 个整数")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导入 AcousticField 的 32 档相位标定结果。"
    )
    parser.add_argument("--phase-bins", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--solver-to-order",
        default="identity",
        help="identity，或包含 256 个 solver_index -> orderNumber 整数的文本文件。",
    )
    parser.add_argument(
        "--mark-verified",
        action="store_true",
        help="仅在 AcousticField 几何聚焦复核通过后使用。",
    )
    args = parser.parse_args()

    bins = read_integer_vector(args.phase_bins, "相位档位文件")
    if np.any(bins < 0) or np.any(bins >= DIVISIONS):
        parser.error("相位档位必须位于 0..31")
    if args.solver_to_order == "identity":
        order = np.arange(CHANNELS, dtype=np.int64)
    else:
        order = read_integer_vector(Path(args.solver_to_order), "通道顺序文件")
        if set(order.tolist()) != set(range(CHANNELS)):
            parser.error("通道顺序必须是 0..255 的完整排列")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        parser.error("目标文件不是有效的 Ultraino YAML 配置")
    config["phase_corrections_pi"] = (
        2.0 * bins.astype(np.float64) / DIVISIONS
    ).tolist()
    config["solver_to_ultraino_order"] = order.tolist()
    config["calibration_source"] = {
        "tool": "AcousticField AssignTransducers export",
        "phase_bins_file": args.phase_bins.name,
        "phase_divisions": DIVISIONS,
    }
    config["calibration_verified"] = bool(args.mark_verified)
    args.config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已写入 {args.config}")


if __name__ == "__main__":
    main()
