"""Plot hand phase surrogate training history from metrics.jsonl.

The training script writes one JSON object per epoch. This utility keeps the
last record for each epoch, then exports compact PNG/SVG plots for loss and
validation metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=str(ROOT / "outputs/hand_phase_surrogate_mvp_train_v1"),
        help="Training output directory containing metrics.jsonl",
    )
    parser.add_argument("--metrics-file", default="", help="Explicit metrics.jsonl path")
    parser.add_argument("--output", default="", help="Output image path; defaults to run_dir/training_curves.png")
    parser.add_argument("--svg", action="store_true", help="Also write an SVG next to the PNG")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    by_epoch: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            epoch = int(record["epoch"])
            by_epoch[epoch] = record

    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _series(records: list[dict[str, Any]], split: str, key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        values.append(float(record.get(split, {}).get(key, "nan")))
    return values


def plot_metrics(records: list[dict[str, Any]], output: Path, *, dpi: int, write_svg: bool) -> None:
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("No metric records to plot")

    epochs = [int(record["epoch"]) for record in records]
    best_index = min(range(len(records)), key=lambda index: float(records[index]["validation"]["total"]))
    best_epoch = epochs[best_index]
    best_val = float(records[best_index]["validation"]["total"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0][0]
    ax.plot(epochs, _series(records, "train", "total"), label="train total", linewidth=1.8)
    ax.plot(epochs, _series(records, "validation", "total"), label="validation total", linewidth=1.8)
    ax.axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.scatter([best_epoch], [best_val], color="tab:red", s=28, zorder=4, label=f"best val @ {best_epoch}")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[0][1]
    ax.plot(epochs, _series(records, "train", "complex_nmse"), label="train NMSE", linewidth=1.8)
    ax.plot(epochs, _series(records, "validation", "complex_nmse"), label="validation NMSE", linewidth=1.8)
    ax.set_title("Complex NMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMSE")
    ax.legend()

    ax = axes[1][0]
    ax.plot(epochs, _series(records, "train", "phase_mae_rad"), label="train phase MAE", linewidth=1.8)
    ax.plot(epochs, _series(records, "validation", "phase_mae_rad"), label="validation phase MAE", linewidth=1.8)
    ax.set_title("Phase Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Radians")
    ax.legend()

    ax = axes[1][1]
    ax.plot(epochs, _series(records, "train", "amplitude_relative_error"), label="train amplitude error", linewidth=1.8)
    ax.plot(
        epochs,
        _series(records, "validation", "amplitude_relative_error"),
        label="validation amplitude error",
        linewidth=1.8,
    )
    ax.set_title("Amplitude Relative Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Error")
    ax.legend()

    fig.suptitle(f"Hand Phase Surrogate Training Curves ({len(records)} epochs)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    if write_svg:
        fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = Path(args.metrics_file) if args.metrics_file else run_dir / "metrics.jsonl"
    output = Path(args.output) if args.output else run_dir / "training_curves.png"
    records = _read_metrics(metrics_path)
    plot_metrics(records, output, dpi=args.dpi, write_svg=args.svg)
    best = min(records, key=lambda record: float(record["validation"]["total"]))
    print(
        json.dumps(
            {
                "records": len(records),
                "first_epoch": int(records[0]["epoch"]),
                "last_epoch": int(records[-1]["epoch"]),
                "best_epoch": int(best["epoch"]),
                "best_validation_total": float(best["validation"]["total"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
