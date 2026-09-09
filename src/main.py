"""Run a configured phase algorithm or a fixed equal-phase forward experiment."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from config import SimulationConfig
from baselines import ALGORITHMS, run, validate_algorithm


def scene_metadata(cfg, solver):
    output = {
        "source_positions": solver.transducers.centers,
        "target_points": np.asarray(cfg.targets).reshape(-1, 3),
        "boundary_conditions": json.dumps(cfg.boundary_conditions),
        "transducer_radius": cfg.specs.diameter * 0.5,
        "dx": cfg.domain.dx, "frequency": cfg.frequency,
        "c0": cfg.physics.sound_speed, "mode": cfg.mode,
    }
    if solver.obstacle_sdf is not None:
        output["sdf"] = solver.obstacle_sdf
    elif cfg.obstacles and cfg.obstacles[0].get("type") == "sphere":
        obs = cfg.obstacles[0]
        axes = [np.linspace(0, length, count)
                for length, count in zip(cfg.domain.box_size, cfg.domain.grid_size)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        output["sdf"] = np.linalg.norm(grid - obs["center"], axis=-1) - obs["radius"]
    return output


def save_phase_optimization_result(
    cfg,
    solver,
    basis,
    problem,
    result,
    basis_setup_seconds: float,
    extra_metadata: dict | None = None,
):
    reporting_started = time.perf_counter()
    initial_field = basis.field(problem.initial_phases, return_numpy=True)
    reporting_evaluations = 1
    output = scene_metadata(cfg, solver)
    output.update(
        algorithm=cfg.algorithm,
        phases=result.final.phases, u_complex=result.final.field,
        amplitude=np.abs(result.final.field), amp_sq=np.abs(result.final.field)**2,
        best_phases=result.best.phases, best_amplitude=np.abs(result.best.field),
        initial_amplitude=np.abs(initial_field),
        target_amplitude=problem.target.target, target_weight=problem.target.weight,
        loss_history=np.array([row["loss"] for row in result.history]),
        best_loss_history=np.array([row["best_loss"] for row in result.history]),
        gradient_norm_history=np.array([
            np.nan if row["gradient_norm"] is None else row["gradient_norm"]
            for row in result.history]),
        evaluation_history=np.array([row["evaluation"] for row in result.history]),
        elapsed_seconds_history=np.array([
            row["elapsed_seconds"] for row in result.history]),
    )
    if cfg.training.compare_geometric:
        phases = solver.transducers.compute_geometric_phases()
        geometric_field = basis.field(phases, return_numpy=True)
        reporting_evaluations += 1
        output.update(
            geometric_phases=phases, geometric_u_complex=geometric_field,
            geometric_amplitude=np.abs(geometric_field),
            geometric_amp_sq=np.abs(geometric_field)**2,
            geometric_metrics=json.dumps(
                problem.metrics(geometric_field), allow_nan=False
            ),
        )
    path = Path(cfg.io.output_file)
    if path.suffix != ".npz":
        raise ValueError("io.output_file must end in .npz")
    loss_plot_path = path.with_name(f"{path.stem}_loss.png")
    metadata = {
        "algorithm": cfg.algorithm, "algorithm_options": cfg.algorithm_options,
        "seed": cfg.training.seed, "termination_reason": result.termination_reason,
        "final_loss": result.final.loss, "best_loss": result.best.loss,
        "final_metrics": result.final.metrics, "best_metrics": result.best.metrics,
        "basis_build_or_load_seconds": basis_setup_seconds,
        "reporting_field_evaluations": reporting_evaluations,
        "reporting_seconds": time.perf_counter() - reporting_started,
        "loss_plot_file": str(loss_plot_path),
        **result.metadata,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    output["run_metadata"] = json.dumps(metadata, allow_nan=False)
    output["evaluation_log"] = json.dumps(result.history, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    from baselines.loss_curve import save_loss_history
    save_loss_history(
        result.history,
        loss_plot_path,
        f"{cfg.algorithm}: current and best loss",
    )
    print(f"[Main] Saved {path}")
    return output, metadata


def execute(cfg):
    if cfg.mode == "sdf_inverse":
        raise NotImplementedError(
            "sdf_inverse is reserved for fixed equal-phase SDF reconstruction. "
            "Observation loading and differentiable geometry assembly are not implemented."
        )
    if cfg.mode == "phase_optimization":
        validate_algorithm(cfg)
    from solvers.helmholtz_solver import HelmholtzDirectSolver

    solver = HelmholtzDirectSolver(cfg)
    basis = None
    started = time.perf_counter()
    try:
        if cfg.mode == "same_phase":
            solver.set_phases(np.full(solver.transducers.num_transducers, cfg.same_phase_rad))
            field = solver.solve()
            output = scene_metadata(cfg, solver)
            output.update(phases=solver.transducers.phases, u_complex=field,
                          amplitude=np.abs(field), amp_sq=np.abs(field)**2)
        else:
            basis = solver.build_phase_response_basis()
            if cfg.training.load_basis_to_gpu:
                basis.to_gpu()
            setup_seconds = time.perf_counter() - started
            problem, result = run(cfg, basis)
            output, _ = save_phase_optimization_result(
                cfg, solver, basis, problem, result, setup_seconds
            )
            print(f"[Result] {cfg.algorithm}: {result.termination_reason}; "
                  f"final={result.final.loss:.6e} best={result.best.loss:.6e}")
        path = Path(cfg.io.output_file)
        if path.suffix != ".npz":
            raise ValueError("io.output_file must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **output)
        print(f"[Main] Saved {path}")
    finally:
        if basis is not None:
            basis.close()
        if solver.condensed_solver is not None:
            solver.condensed_solver.close()
    if cfg.io.auto_visualize:
        from visualizer import show_pyvista_scene
        show_pyvista_scene(
            cfg.io.output_file, field_name="amplitude",
            compare_methods=cfg.mode == "phase_optimization" and cfg.training.compare_geometric,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Ultrasound phase optimization and SDF experiments")
    parser.add_argument("--config", default="src/examples/config.yaml")
    parser.add_argument("--list-algorithms", action="store_true")
    args = parser.parse_args()
    if args.list_algorithms:
        print("\n".join(ALGORITHMS))
        return
    execute(SimulationConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
