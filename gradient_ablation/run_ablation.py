"""Compare phase update rules under one shared analytic-gradient problem."""

import argparse
import copy
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from baselines import run, validate_algorithm
from baselines.loss_curve import ComparisonLossDashboard, save_loss_dashboard
from config import SimulationConfig
from main import save_phase_optimization_result
from solvers.helmholtz_solver import HelmholtzDirectSolver


OPTIMIZERS = {
    "lbfgsb": {
        "optimizer": "lbfgsb",
        "iterations": 5000,
        "max_evaluations": 15000,
    },
    "adam": {
        "optimizer": "adam",
        "learning_rate": 0.05,
        "convergence_patience": 100,
        "convergence_relative_tolerance": 1.0e-5,
        "iterations": 5000,
        "max_evaluations": 15000,
    },
    "adamw": {
        "optimizer": "adamw",
        "learning_rate": 0.05,
        # Phase is periodic; nonzero Euclidean weight decay is not physical.
        "weight_decay": 0.0,
        "convergence_patience": 100,
        "convergence_relative_tolerance": 1.0e-5,
        "iterations": 5000,
        "max_evaluations": 15000,
    },
    "lion": {
        "optimizer": "lion",
        "learning_rate": 0.02,
        "beta1": 0.9,
        "beta2": 0.99,
        "convergence_patience": 100,
        "convergence_relative_tolerance": 1.0e-5,
        "iterations": 5000,
        "max_evaluations": 15000,
    },
    "nonlinear_cg": {
        "optimizer": "nonlinear_cg",
        "initial_step": 0.25,
        "armijo": 1.0e-4,
        "line_search_shrink": 0.5,
        "max_line_search": 12,
        "convergence_patience": 100,
        "convergence_relative_tolerance": 1.0e-5,
        "iterations": 5000,
        "max_evaluations": 15000,
    },
}

QUALITY_TARGETS = (
    ("lbfgsb_quality", -1593.498),
    ("strong_quality", -1800.0),
    ("adam_quality", -1860.0),
)


def _portable_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _config(base_cfg, name, output_dir, show_loss_curve, max_seconds):
    profile = OPTIMIZERS[name]
    cfg = copy.deepcopy(base_cfg)
    cfg.algorithm = "adjoint"
    cfg.algorithm_options = {
        key: value
        for key, value in profile.items()
        if key not in {"iterations", "max_evaluations"}
    }
    cfg.training.initial_phase = "geometric"
    cfg.training.iterations = profile["iterations"]
    cfg.training.max_evaluations = profile["max_evaluations"]
    cfg.training.max_seconds = max_seconds
    cfg.training.show_loss_curve = show_loss_curve
    cfg.io.auto_visualize = False
    cfg.io.output_file = _portable_path(output_dir / name / "result.npz")
    return cfg


def _summary_row(name, status, metadata=None, error=""):
    row = {"optimizer": name, "status": status, "error": error}
    if metadata:
        metrics = metadata["best_metrics"]
        row.update(
            best_loss=metadata["best_loss"],
            final_loss=metadata["final_loss"],
            target_mean=metrics["target_mean"],
            background_max=metrics["background_max"],
            contrast=metrics["contrast"],
            field_evaluations=metadata["field_evaluations"],
            vjp_evaluations=metadata["vjp_evaluations"],
            optimization_seconds=metadata["elapsed_seconds"],
            termination_reason=metadata["termination_reason"],
        )
    return row


def _write_summary(output_dir, config_path, basis_seconds, rows):
    config_path = Path(config_path).resolve()
    try:
        config_label = str(config_path.relative_to(ROOT))
    except ValueError:
        config_label = config_path.name
    payload = {
        "source_config": config_label,
        "shared_basis_build_or_load_seconds": basis_seconds,
        "optimizers": list(OPTIMIZERS),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_quality_summary(output_dir, histories):
    rows = []
    for optimizer, history in histories.items():
        for target_name, threshold in QUALITY_TARGETS:
            reached = next(
                (
                    row
                    for row in history
                    if row["best_loss"] <= threshold
                ),
                None,
            )
            rows.append(
                {
                    "optimizer": optimizer,
                    "quality_target": target_name,
                    "loss_threshold": threshold,
                    "reached": reached is not None,
                    "field_evaluations_to_target": (
                        reached["evaluation"] if reached is not None else None
                    ),
                    "seconds_to_target": (
                        reached["elapsed_seconds"] if reached is not None else None
                    ),
                }
            )
    payload = {
        "quality_targets": [
            {"name": name, "loss_threshold": threshold}
            for name, threshold in QUALITY_TARGETS
        ],
        "rows": rows,
    }
    (output_dir / "quality_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "quality_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "optimizer",
                "quality_target",
                "loss_threshold",
                "reached",
                "field_evaluations_to_target",
                "seconds_to_target",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def rebuild_quality_summary(output_dir):
    histories = {}
    for path in Path(output_dir).glob("*/result.npz"):
        with np.load(path) as result:
            histories[path.parent.name] = [
                {
                    "evaluation": int(evaluation),
                    "elapsed_seconds": float(elapsed_seconds),
                    "best_loss": float(best_loss),
                }
                for evaluation, elapsed_seconds, best_loss in zip(
                    result["evaluation_history"],
                    result["elapsed_seconds_history"],
                    result["best_loss_history"],
                )
            ]
    _write_quality_summary(Path(output_dir), histories)


def run_ablation(config_path, output_root, optimizers, show_loss_curve, max_seconds):
    base_cfg = SimulationConfig.from_yaml(config_path)
    if base_cfg.mode != "phase_optimization":
        raise ValueError("Gradient ablation requires mode: phase_optimization")
    unknown = [name for name in optimizers if name not in OPTIMIZERS]
    if unknown:
        raise ValueError(f"Unknown optimizers: {unknown}")

    output_dir = Path(output_root) / Path(config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard = ComparisonLossDashboard(
        optimizers,
        enabled=show_loss_curve,
        pause_seconds=base_cfg.training.loss_curve_pause_seconds,
        tensorboard_log_dir=output_dir / "tensorboard",
    )
    solver = HelmholtzDirectSolver(base_cfg)
    basis = None
    histories, rows = {}, []
    started = time.perf_counter()
    try:
        basis = solver.build_phase_response_basis()
        if base_cfg.training.load_basis_to_gpu:
            basis.to_gpu()
        basis_seconds = time.perf_counter() - started
        for name in optimizers:
            cfg = _config(
                base_cfg, name, output_dir, show_loss_curve, max_seconds
            )
            method_dir = output_dir / name
            method_dir.mkdir(parents=True, exist_ok=True)
            (method_dir / "config.yaml").write_text(
                yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            print(f"\n{'=' * 72}\n[GradientAblation] Running {name}\n{'=' * 72}")
            try:
                validate_algorithm(cfg)
                problem, result = run(
                    cfg,
                    basis,
                    loss_curve=dashboard.curve(
                        name, cfg.training.loss_curve_update_interval
                    ),
                )
                _, metadata = save_phase_optimization_result(
                    cfg,
                    solver,
                    basis,
                    problem,
                    result,
                    0.0,
                    extra_metadata={
                        "gradient_ablation_optimizer": name,
                        "gradient_ablation_profile": OPTIMIZERS[name],
                        "shared_basis_build_or_load_seconds": basis_seconds,
                    },
                )
                histories[name] = result.history
                rows.append(_summary_row(name, "completed", metadata))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rows.append(_summary_row(name, "failed", error=error))
                (method_dir / "error.txt").write_text(
                    f"{error}\n\n{traceback.format_exc()}", encoding="utf-8"
                )
                print(f"[GradientAblation] {name} failed: {error}")
            _write_summary(output_dir, config_path, basis_seconds, rows)
    finally:
        save_loss_dashboard(histories, output_dir / "loss_dashboard.png")
        _write_quality_summary(output_dir, histories)
        dashboard.close()
        if basis is not None:
            basis.close()
        if solver.condensed_solver is not None:
            solver.condensed_solver.close()
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Ablate phase update rules with one shared analytic gradient."
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "gradient_ablation" / "config"
            / "phase_oblique_reflecting_x_12x12.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "gradient_ablation" / "results")
    )
    parser.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    parser.add_argument("--no-loss-window", action="store_true")
    parser.add_argument(
        "--rebuild-quality-summary",
        action="store_true",
        help="Regenerate quality summary CSV/JSON from existing result files.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=60.0,
        help="Per-optimizer wall-clock budget; 0 disables the time limit.",
    )
    args = parser.parse_args()
    if args.rebuild_quality_summary:
        output_dir = Path(args.output_dir) / Path(args.config).stem
        rebuild_quality_summary(output_dir)
        print(f"[GradientAblation] rebuilt quality summary: {output_dir}")
        return
    optimizers = [item.strip() for item in args.optimizers.split(",") if item.strip()]
    rows = run_ablation(
        args.config,
        args.output_dir,
        optimizers,
        show_loss_curve=not args.no_loss_window,
        max_seconds=args.max_seconds,
    )
    completed = sum(row["status"] == "completed" for row in rows)
    print(f"[GradientAblation] completed {completed}/{len(rows)} optimizers")
    if completed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
