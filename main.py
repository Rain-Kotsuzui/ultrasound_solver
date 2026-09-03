import argparse
import json
import numpy as np
from config import SimulationConfig
from geometry_sdf import SDFScene
from helmholtz_solver import HelmholtzDirectSolver
from visualizer import show_pyvista_scene


def evaluate_targets(amplitude_field: np.ndarray, targets: list, dx: float):
    pressures = []
    print("\n" + "="*65)
    print("【多焦点声压与均匀度评测指标】")
    for idx, pt in enumerate(targets):
        i = int(round(pt[0] / dx))
        j = int(round(pt[1] / dx))
        k = int(round(pt[2] / dx))
        p_val = amplitude_field[i, j, k]
        pressures.append(p_val)
        print(f"  --> 焦点 {idx+1} {pt}: 声压幅值 = {p_val:.2f} Pa")

    pressures = np.array(pressures)
    mean_p = np.mean(pressures)
    min_p = np.min(pressures)
    max_p = np.max(pressures)
    
    uniformity = (min_p / max_p) if max_p > 0 else 0.0
    cv = (np.std(pressures) / mean_p) * 100 if mean_p > 0 else 0.0

    print(f"  [统计] 平均声压: {mean_p:.2f} Pa | 最小: {min_p:.2f} Pa | 最大: {max_p:.2f} Pa")
    print(f"  [指标] 均匀度 (Min/Max Ratio) = {uniformity:.4f} (越接近 1.0 越均等)")
    print(f"  [指标] 变异系数 (CV) = {cv:.2f}% (越小越好)")
    print("="*65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Warp-Accelerated 3D Ultrasound Forward Solver")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    # 1. 解析配置
    cfg = SimulationConfig.from_yaml(args.config)

    print("="*65)
    print(f"换能器规格: {cfg.frequency:.1f} Hz, SPL={cfg.specs.spl_db:.1f} dB @ {cfg.specs.spl_distance:.3f} m, 直径={cfg.specs.diameter:.4f} m")
    print(f"--> 标定发射表面声压 P0 = {cfg.calibrated_surface_pressure:.2f} Pa (波长 λ = {cfg.wavelength*1000:.2f} mm)")
    print(f"--> 计算区域尺寸: [{cfg.lx*100:.1f} x {cfg.ly*100:.1f} x {cfg.lz*100:.1f}] cm, 网格步长 dx = {cfg.domain.dx*1000:.3f} mm")
    print("="*65)

    # 2. 求解声场
    solver = HelmholtzDirectSolver(cfg)

    if cfg.mode == "baseline":
        u_field = solver.solve()
        amplitude_field = np.abs(u_field)

        # 3. 评测多焦点
        evaluate_targets(amplitude_field, cfg.targets, cfg.domain.dx)

        # 4. 生成空间 SDF 标量场
        nx, ny, nz = cfg.domain.nx, cfg.domain.ny, cfg.domain.nz
        dx = cfg.domain.dx
        xs = np.linspace(0.0, (nx - 1) * dx, nx)
        ys = np.linspace(0.0, (ny - 1) * dx, ny)
        zs = np.linspace(0.0, (nz - 1) * dx, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        scene = SDFScene(cfg.obstacles)
        p = np.stack([X, Y, Z], axis=-1)
        sdf_grid = np.full(X.shape, np.inf)
        if cfg.obstacles:
            obs = cfg.obstacles[0]
            center = np.array(obs.get("center", [0.04, 0.04, 0.02]), dtype=np.float64)
            rad = float(obs.get("radius", 0.012))
            sdf_grid = np.linalg.norm(p - center, axis=-1) - rad

        # 5. 保存完整物理元数据
        out_file = cfg.io.output_file
        np.savez_compressed(
            out_file,
            amplitude=amplitude_field,
            amp_sq=amplitude_field**2,
            u_complex=u_field,
            sdf=sdf_grid,
            source_positions=solver.transducers.centers,
            target_points=np.array(cfg.targets),
            boundary_conditions=json.dumps(cfg.boundary_conditions),
            transducer_radius=cfg.specs.diameter * 0.5,
            dx=cfg.domain.dx,
            frequency=cfg.frequency,
            c0=cfg.physics.sound_speed
        )
        print(f"[Main] 仿真结果及 3D 元数据已保存至: '{out_file}'")

        # 6. 唤起 PyVista 硬件加速可视化
        if cfg.io.auto_visualize:
            show_pyvista_scene(out_file)

    elif cfg.mode == "inverse":
        training_mode = cfg.training.mode.lower()
        if training_mode == "phase_only":
            print("[Main] Building/loading phase-only response basis...")
            basis = solver.build_phase_response_basis()
            if cfg.training.load_basis_to_gpu:
                basis.to_gpu()
            location = "GPU" if basis.basis_gpu is not None else "CPU"
            print(
                "[Main] Phase-only training forward model ready | "
                f"shape={basis.shape} | storage={location}"
            )
        elif training_mode == "obstacle_distribution":
            print(
                "[Main] Obstacle-distribution training selected. "
                "The matrix changes with the obstacle field, so the "
                "phase response basis is intentionally disabled."
            )
        else:
            raise ValueError(
                "training.mode must be 'phase_only' or "
                "'obstacle_distribution'"
            )


if __name__ == "__main__":
    main()
