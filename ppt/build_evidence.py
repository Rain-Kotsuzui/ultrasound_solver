"""Export traceable numerical evidence for the offline defense slides."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "results/phase_oblique_reflecting_x_12x12"
ALGORITHMS = ROOT / "algorithm_comparison" / ARCHIVE
ABLATION = ROOT / "gradient_ablation" / ARCHIVE


def read_quality(path, key):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [
            {
                "id": row[key],
                "threshold": float(row["loss_threshold"]),
                "reached": row["reached"] == "True",
                "seconds": float(row["seconds_to_target"])
                if row["seconds_to_target"] else None,
                "evaluations": int(row["field_evaluations_to_target"])
                if row["field_evaluations_to_target"] else None,
            }
            for row in csv.DictReader(stream)
        ]


def main():
    source = ALGORITHMS / "adjoint/result.npz"
    with np.load(source, allow_pickle=False) as result:
        dx = float(result["dx"])
        target = result["target_points"][0]
        iy = int(round(target[1] / dx))
        first_z = 4
        optimized = result["best_amplitude"][:, iy, first_z:].T
        geometric = result["geometric_amplitude"][:, iy, first_z:].T
        common_max = float(max(optimized.max(), geometric.max()))
        x = np.arange(optimized.shape[1]) * dx * 1000
        z = np.arange(first_z, first_z + optimized.shape[0]) * dx * 1000
        figure, ax = plt.subplots()
        contours = ax.contour(
            x, z, optimized / common_max,
            levels=[0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95],
        )
        paths = [
            {"level": float(level), "points": segment.round(3).tolist()}
            for level, segments in zip(contours.levels, contours.allsegs)
            for segment in segments if len(segment) > 2
        ]
        plt.close(figure)
        field = {
            "source": source.relative_to(ROOT).as_posix(),
            "plane_y_mm": round(iy * dx * 1000, 3),
            "x_mm": [float(x[0]), float(x[-1])],
            "z_mm": [float(z[0]), float(z[-1])],
            "target_mm": (target * 1000).round(3).tolist(),
            "normalization": "common maximum of both displayed slices",
            "normalization_value": common_max,
            "optimized": (optimized / common_max).round(6).tolist(),
            "geometric": (geometric / common_max).round(6).tolist(),
            "contours": paths,
        }
    data = {
        "algorithm_source": (ALGORITHMS / "quality_summary.csv").relative_to(ROOT).as_posix(),
        "ablation_source": (ABLATION / "quality_summary.csv").relative_to(ROOT).as_posix(),
        "protocol": {"array": [12, 12], "grid": [41, 41, 41], "seed": 0,
                     "seconds_budget": 60, "field_budget": 15000},
        "algorithms": read_quality(ALGORITHMS / "quality_summary.csv", "algorithm"),
        "ablation": read_quality(ABLATION / "quality_summary.csv", "optimizer"),
        "field": field,
    }
    destination = ROOT / "ppt/assets/evidence.js"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "window.DEFENSE_EVIDENCE = "
        + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + ";\n", encoding="utf-8",
    )
    print(f"Exported {len(paths)} contours and quality records to {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
