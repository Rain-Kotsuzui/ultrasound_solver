"""Run and analyze the three-obstacle lateral-displacement study."""

from __future__ import annotations

import csv
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import yaml


ROOT = Path(__file__).resolve().parents[1]
STUDY = Path(__file__).resolve().parent
MODELS = STUDY / "models"
CONFIGS = STUDY / "configs"
RESULTS = STUDY / "results"
TOP_FIELDS = STUDY / "top_fields"
FIGURES = STUDY / "figures"
LOGS = STUDY / "logs"

DOMAIN_SIZE_M = 0.100
GRID_N = 70
GRID_SPACING_M = DOMAIN_SIZE_M / (GRID_N - 1)
LEFT_X_M = 0.0485
RIGHT_X_M = 0.0515
CENTER_Y_M = 0.0500
MATERIAL = {"density": 1250.0, "sound_speed": 2200.0}

CASES = (
    {
        "key": "plate",
        "label": "Plate",
        "mesh": "plate_22x22x3mm.vtk",
        "center_z": 0.045,
        "rotation_deg": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "dimensions_mm": "22.0 x 22.0 x 3.0",
    },
    {
        "key": "sphere",
        "label": "Sphere",
        "mesh": "sphere_d20mm.vtk",
        "center_z": 0.045,
        "rotation_deg": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "dimensions_mm": "20.0 x 20.0 x 20.0",
    },
    {
        "key": "hand",
        "label": "Hand",
        "mesh": "hand_watertight.vtk",
        "center_z": 0.048,
        "rotation_deg": [0.0, 90.0, 0.0],
        "scale": 0.18,
        "dimensions_mm": "49.8 x 20.0 x 15.4",
    },
)

POSITIONS = (
    ("left", LEFT_X_M),
    ("right", RIGHT_X_M),
)


def ensure_directories() -> None:
    for directory in (MODELS, CONFIGS, RESULTS, TOP_FIELDS, FIGURES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def create_models() -> None:
    plate = pv.Box(bounds=(-0.011, 0.011, -0.011, 0.011, -0.0015, 0.0015))
    plate.triangulate().save(MODELS / "plate_22x22x3mm.vtk")

    sphere = pv.Sphere(radius=0.010, theta_resolution=48, phi_resolution=48)
    sphere.triangulate().save(MODELS / "sphere_d20mm.vtk")

    source_hand = ROOT / "mesh_hand_result" / "hand_watertight.vtk"
    if not source_hand.is_file():
        raise FileNotFoundError(f"Missing hand mesh: {source_hand}")
    shutil.copy2(source_hand, MODELS / "hand_watertight.vtk")


def case_config(case: dict, position_name: str, center_x: float) -> dict:
    output = RESULTS / f"{case['key']}_{position_name}.npz"
    return {
        "mode": "baseline",
        "physics": {"sound_speed": 343.0, "medium_density": 1.21},
        "transducer_specs": {
            "frequency": 40000.0,
            "spl_db": 115.0,
            "spl_distance": 0.3,
            "diameter": 0.01,
            "pitch": 0.01,
            "array_n": 9,
        },
        "domain": {
            "box_size": [DOMAIN_SIZE_M] * 3,
            "grid_size": [GRID_N] * 3,
        },
        "solver": {
            "backend": "cudss_hybrid_direct",
            "direct_residual_tol": 5.0e-4,
            "hybrid_num_threads": 1,
            "hybrid_execution_mode": "auto",
            "hybrid_device_memory_limit_gb": 5.6,
            "hybrid_device_memory_reserve_gb": 0.75,
            "hybrid_register_cuda_memory": False,
            "condensed_symmetry_tolerance": 1.0e-12,
        },
        "boundary_conditions": {
            "-x": "open",
            "+x": "open",
            "-y": "open",
            "+y": "open",
            "-z": "reflecting",
            "+z": "open",
        },
        # An empty target list makes TransducerArray prescribe phase 0 to all emitters.
        "targets": [],
        "obstacles": [
            {
                "type": "mesh",
                "file": str((MODELS / case["mesh"]).resolve()),
                "center_m": [center_x, CENTER_Y_M, case["center_z"]],
                "rotation_deg": case["rotation_deg"],
                "scale": case["scale"],
                "smoothing_width_cells": 1.2,
                "material": MATERIAL,
            }
        ],
        "io": {"output_file": str(output.resolve()), "auto_visualize": False},
    }


def write_configs() -> list[tuple[dict, str, Path]]:
    jobs = []
    for case in CASES:
        for position_name, center_x in POSITIONS:
            config_path = CONFIGS / f"{case['key']}_{position_name}.yaml"
            config = case_config(case, position_name, center_x)
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            jobs.append((case, position_name, config_path))
    return jobs


def run_jobs(jobs: list[tuple[dict, str, Path]]) -> None:
    for index, (case, position_name, config_path) in enumerate(jobs, start=1):
        output = RESULTS / f"{case['key']}_{position_name}.npz"
        if output.is_file():
            print(f"[{index}/6] Reusing {output.name}")
            continue
        print(f"[{index}/6] Solving {case['label']} at {position_name} position")
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "main.py"),
                "--config",
                str(config_path),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (LOGS / f"{case['key']}_{position_name}.log").write_text(
            process.stdout, encoding="utf-8"
        )
        if process.returncode != 0:
            print(process.stdout)
            raise RuntimeError(f"Simulation failed: {case['key']}_{position_name}")
        gc.collect()


def save_csv(path: Path, field: np.ndarray) -> None:
    np.savetxt(path, field, delimiter=",", fmt="%.9e")


def load_result(
    case_key: str,
    position_name: str,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    path = RESULTS / f"{case_key}_{position_name}.npz"
    with np.load(path) as data:
        top_amplitude = np.asarray(data["amplitude"][:, :, -1], dtype=np.float64)
        sdf = np.asarray(data["sdf"], dtype=np.float64)
        dx = float(data["dx"])
        source_positions = np.asarray(data["source_positions"], dtype=np.float64)
        transducer_radius = float(data["transducer_radius"])
    return top_amplitude, sdf, dx, source_positions, transducer_radius


def create_top_field_figure(case: dict, left: np.ndarray, right: np.ndarray) -> None:
    signed_diff = right - left
    field_min = min(float(left.min()), float(right.min()))
    field_max = max(float(left.max()), float(right.max()))
    diff_limit = float(np.max(np.abs(signed_diff)))
    extent = [0.0, 100.0, 0.0, 100.0]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    image_left = axes[0].imshow(
        left.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
        interpolation="nearest",
    )
    axes[0].set_title("Left position (x = 48.5 mm)")
    axes[1].imshow(
        right.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
        interpolation="nearest",
    )
    axes[1].set_title("Right position (x = 51.5 mm)")
    image_diff = axes[2].imshow(
        signed_diff.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-diff_limit,
        vmax=diff_limit,
        interpolation="nearest",
    )
    axes[2].set_title("Signed difference (right - left)")
    for axis in axes:
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_aspect("equal")
    fig.colorbar(image_left, ax=axes[:2], label="Amplitude (Pa)", shrink=0.88)
    fig.colorbar(image_diff, ax=axes[2], label="Difference (Pa)", shrink=0.88)
    fig.suptitle(
        f"{case['label']} obstacle: top-plane amplitude at z = 100 mm\n"
        "All 81 emitter phases fixed at 0 rad"
    )
    fig.savefig(FIGURES / f"{case['key']}_top_comparison.png", dpi=180)
    plt.close(fig)


def create_top_field_summary(rows: list[dict]) -> None:
    field_min = min(
        min(float(row["left"].min()), float(row["right"].min()))
        for row in rows
    )
    field_max = max(
        max(float(row["left"].max()), float(row["right"].max()))
        for row in rows
    )
    diff_limit = max(
        float(np.max(np.abs(row["right"] - row["left"])))
        for row in rows
    )
    extent = [0.0, 100.0, 0.0, 100.0]
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(14.5, 13.2),
        constrained_layout=True,
    )

    field_image = None
    diff_image = None
    for row_index, row in enumerate(rows):
        signed_diff = row["right"] - row["left"]
        field_image = axes[row_index, 0].imshow(
            row["left"].T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
            interpolation="nearest",
        )
        axes[row_index, 1].imshow(
            row["right"].T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
            interpolation="nearest",
        )
        diff_image = axes[row_index, 2].imshow(
            signed_diff.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-diff_limit,
            vmax=diff_limit,
            interpolation="nearest",
        )
        axes[row_index, 0].set_ylabel(f"{row['case']['label']}\ny (mm)")
        for column in range(3):
            axes[row_index, column].set_xlabel("x (mm)")
            axes[row_index, column].set_aspect("equal")

    axes[0, 0].set_title("Left: x = 48.5 mm")
    axes[0, 1].set_title("Right: x = 51.5 mm")
    axes[0, 2].set_title("Difference: right - left")
    fig.colorbar(
        field_image,
        ax=axes[:, :2].ravel().tolist(),
        label="Amplitude (Pa)",
        shrink=0.72,
    )
    fig.colorbar(
        diff_image,
        ax=axes[:, 2].ravel().tolist(),
        label="Difference (Pa)",
        shrink=0.72,
    )
    fig.suptitle(
        "Top-plane amplitude response to a 3 mm obstacle displacement\n"
        "All 81 emitter phases fixed at 0 rad"
    )
    fig.savefig(FIGURES / "all_top_field_comparisons.png", dpi=180)
    plt.close(fig)


def create_cross_obstacle_differences(rows: list[dict]) -> list[dict]:
    pairs = []
    for base_index in range(len(rows)):
        for compare_index in range(base_index + 1, len(rows)):
            base = rows[base_index]
            compare = rows[compare_index]
            difference = compare["left"] - base["left"]
            pairs.append(
                {
                    "base": base["case"]["label"],
                    "compare": compare["case"]["label"],
                    "base_key": base["case"]["key"],
                    "compare_key": compare["case"]["key"],
                    "difference": difference,
                    "relative_l2": float(
                        np.linalg.norm(difference) / np.linalg.norm(base["left"])
                    ),
                    "symmetric_relative_l2": float(
                        np.linalg.norm(difference)
                        / np.sqrt(
                            np.linalg.norm(base["left"])
                            * np.linalg.norm(compare["left"])
                        )
                    ),
                    "mean_absolute_difference_pa": float(np.mean(np.abs(difference))),
                    "max_absolute_difference_pa": float(np.max(np.abs(difference))),
                    "spatial_correlation": float(
                        np.corrcoef(base["left"].ravel(), compare["left"].ravel())[0, 1]
                    ),
                }
            )

    diff_limit = max(
        float(np.max(np.abs(pair["difference"])))
        for pair in pairs
    )
    extent = [0.0, 100.0, 0.0, 100.0]
    fig, axes = plt.subplots(1, len(pairs), figsize=(16.0, 4.8), constrained_layout=True)
    diff_image = None
    for axis, pair in zip(axes, pairs):
        diff_image = axis.imshow(
            pair["difference"].T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-diff_limit,
            vmax=diff_limit,
            interpolation="nearest",
        )
        axis.set_title(f"{pair['compare']} - {pair['base']}")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_aspect("equal")
        save_csv(
            TOP_FIELDS
            / f"{pair['compare_key']}_minus_{pair['base_key']}_left_top_difference_pa.csv",
            pair["difference"],
        )

    np.savez_compressed(
        TOP_FIELDS / "cross_obstacle_left_top_differences.npz",
        **{
            f"{pair['compare_key']}_minus_{pair['base_key']}": pair["difference"]
            for pair in pairs
        },
        x_mm=np.linspace(0.0, DOMAIN_SIZE_M * 1000.0, GRID_N),
        y_mm=np.linspace(0.0, DOMAIN_SIZE_M * 1000.0, GRID_N),
        z_mm=DOMAIN_SIZE_M * 1000.0,
    )
    metric_rows = [
        {
            key: value
            for key, value in pair.items()
            if key != "difference"
        }
        for pair in pairs
    ]
    with (STUDY / "cross_obstacle_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (STUDY / "cross_obstacle_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metric_rows, stream, indent=2)
    fig.colorbar(
        diff_image,
        ax=axes.ravel().tolist(),
        label="Amplitude difference (Pa)",
        shrink=0.86,
    )
    fig.suptitle(
        "Cross-obstacle top-plane amplitude differences at x = 48.5 mm\n"
        "All 81 emitter phases fixed at 0 rad"
    )
    fig.savefig(FIGURES / "cross_obstacle_top_field_differences.png", dpi=180)
    plt.close(fig)
    return pairs


def sdf_zero_surface(sdf: np.ndarray, dx: float) -> pv.PolyData:
    spacing_mm = dx * 1000.0
    grid = pv.ImageData(
        dimensions=sdf.shape,
        spacing=(spacing_mm, spacing_mm, spacing_mm),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data["sdf"] = sdf.ravel(order="F")
    surface = grid.contour([0.0], scalars="sdf")
    if surface.n_points == 0:
        raise RuntimeError("SDF does not contain a zero level set")
    return surface


def rotation_matrix_degrees(rotation_deg: list[float]) -> np.ndarray:
    angles = np.deg2rad(np.asarray(rotation_deg, dtype=np.float64))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ]
    )
    ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ]
    )
    rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rz @ ry @ rx


def transformed_obstacle_surface(case: dict, center_x: float) -> pv.PolyData:
    mesh = pv.read(MODELS / case["mesh"]).triangulate()
    points_m = np.asarray(mesh.points, dtype=np.float64)
    mesh_center_m = np.asarray(mesh.center, dtype=np.float64)
    rotation = rotation_matrix_degrees(case["rotation_deg"])
    center_m = np.array([center_x, CENTER_Y_M, case["center_z"]], dtype=np.float64)
    transformed_m = (points_m - mesh_center_m) * case["scale"]
    transformed_m = transformed_m @ rotation.T + center_m
    mesh.points = transformed_m * 1000.0
    return mesh.extract_surface().clean()


def transducer_array_surface(
    source_positions: np.ndarray,
    transducer_radius: float,
) -> pv.PolyData:
    radius_mm = transducer_radius * 1000.0
    height_mm = 0.8
    transducers = [
        pv.Cylinder(
            center=(
                position[0] * 1000.0,
                position[1] * 1000.0,
                height_mm * 0.5,
            ),
            direction=(0.0, 0.0, 1.0),
            radius=radius_mm,
            height=height_mm,
            resolution=32,
        )
        for position in source_positions
    ]
    return pv.merge(transducers)


def create_sdf_overlay(rows: list[dict], output_path: Path) -> None:
    plotter = pv.Plotter(
        shape=(1, len(rows)),
        off_screen=True,
        window_size=(900 * len(rows), 760),
        border=False,
    )
    plotter.set_background("#f8f9fa")
    plotter.enable_depth_peeling(
        number_of_peels=8,
        occlusion_ratio=0.0,
    )
    original_color = "#228be6"
    shifted_color = "#fa5252"
    array_color = "#495057"
    camera_position = [
        (165.0, -135.0, 165.0),
        (50.0, 50.0, 45.0),
        (0.0, 0.0, 1.0),
    ]

    for column, row in enumerate(rows):
        original_surface = transformed_obstacle_surface(row["case"], LEFT_X_M)
        shifted_surface = transformed_obstacle_surface(row["case"], RIGHT_X_M)
        array_surface = transducer_array_surface(
            row["source_positions"],
            row["transducer_radius"],
        )

        plotter.subplot(0, column)
        plotter.add_mesh(
            array_surface,
            color=array_color,
            smooth_shading=True,
            specular=0.1,
        )
        plotter.add_mesh(
            original_surface,
            color=original_color,
            opacity=0.42,
            smooth_shading=True,
            specular=0.1,
        )
        plotter.add_mesh(
            shifted_surface,
            color=shifted_color,
            opacity=0.42,
            smooth_shading=True,
            specular=0.1,
        )
        plotter.add_text(
            f"{row['case']['label']} | shift: +x 3 mm\n"
            "Blue: original\n"
            "Red: shifted",
            position="upper_left",
            font_size=11,
            color="black",
        )
        plotter.show_bounds(
            bounds=(0.0, 100.0, 0.0, 100.0, 0.0, 100.0),
            location="outer",
            all_edges=True,
            xtitle="x (mm)",
            ytitle="y (mm)",
            ztitle="z (mm)",
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
            font_size=9,
            color="#495057",
        )
        plotter.camera_position = camera_position
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 82.0

    plotter.screenshot(str(output_path))
    plotter.close()


def analyze() -> None:
    summary_rows = []
    visualization_rows = []
    coordinates_mm = np.linspace(0.0, DOMAIN_SIZE_M * 1000.0, GRID_N)

    for case in CASES:
        left, left_sdf, left_dx, source_positions, transducer_radius = load_result(
            case["key"],
            "left",
        )
        right, right_sdf, right_dx, right_positions, right_radius = load_result(
            case["key"],
            "right",
        )
        if not np.isclose(left_dx, right_dx):
            raise RuntimeError(f"Grid spacing mismatch for {case['key']}")
        if not np.allclose(source_positions, right_positions):
            raise RuntimeError(f"Transducer position mismatch for {case['key']}")
        if not np.isclose(transducer_radius, right_radius):
            raise RuntimeError(f"Transducer radius mismatch for {case['key']}")

        signed_diff = right - left
        abs_diff = np.abs(signed_diff)
        base_norm = np.linalg.norm(left)
        rel_l2 = np.linalg.norm(signed_diff) / base_norm if base_norm else np.nan
        correlation = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
        rms_diff = float(np.sqrt(np.mean(signed_diff**2)))
        mean_abs = float(np.mean(abs_diff))
        max_abs = float(np.max(abs_diff))

        np.savez_compressed(
            TOP_FIELDS / f"{case['key']}_top_fields.npz",
            left=left,
            right=right,
            signed_difference=signed_diff,
            absolute_difference=abs_diff,
            x_mm=coordinates_mm,
            y_mm=coordinates_mm,
            z_mm=DOMAIN_SIZE_M * 1000.0,
        )
        save_csv(TOP_FIELDS / f"{case['key']}_left_top_amplitude_pa.csv", left)
        save_csv(TOP_FIELDS / f"{case['key']}_right_top_amplitude_pa.csv", right)
        save_csv(TOP_FIELDS / f"{case['key']}_signed_difference_pa.csv", signed_diff)
        save_csv(TOP_FIELDS / f"{case['key']}_absolute_difference_pa.csv", abs_diff)

        create_top_field_figure(case, left, right)
        visualization_row = {
            "case": case,
            "left": left,
            "right": right,
            "left_sdf": left_sdf,
            "right_sdf": right_sdf,
            "dx": left_dx,
            "source_positions": source_positions,
            "transducer_radius": transducer_radius,
        }
        visualization_rows.append(visualization_row)
        create_sdf_overlay(
            [visualization_row],
            FIGURES / f"{case['key']}_sdf_3d_comparison.png",
        )

        summary_rows.append(
            {
                "model": case["label"],
                "dimensions_mm": case["dimensions_mm"],
                "left_center_x_mm": LEFT_X_M * 1000.0,
                "right_center_x_mm": RIGHT_X_M * 1000.0,
                "displacement_mm": (RIGHT_X_M - LEFT_X_M) * 1000.0,
                "top_peak_left_pa": float(np.max(left)),
                "top_peak_right_pa": float(np.max(right)),
                "mean_absolute_difference_pa": mean_abs,
                "rms_difference_pa": rms_diff,
                "max_absolute_difference_pa": max_abs,
                "relative_l2_difference": rel_l2,
                "spatial_correlation": correlation,
                "all_emitter_phases_rad": 0.0,
            }
        )

    summary_path = STUDY / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (STUDY / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary_rows, stream, indent=2)

    create_top_field_summary(visualization_rows)
    create_sdf_overlay(
        visualization_rows,
        FIGURES / "all_sdf_3d_comparisons.png",
    )
    cross_obstacle_rows = create_cross_obstacle_differences(visualization_rows)
    create_summary_figure(summary_rows)
    write_readme(summary_rows, cross_obstacle_rows)


def create_summary_figure(rows: list[dict]) -> None:
    labels = [row["model"] for row in rows]
    relative = [100.0 * row["relative_l2_difference"] for row in rows]
    correlation = [row["spatial_correlation"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = ["#2878B5", "#F39B35", "#C82423"]
    axes[0].bar(labels, relative, color=colors)
    axes[0].set_ylabel("Relative L2 difference (%)")
    axes[0].set_title("Sensitivity to 3 mm lateral displacement")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, correlation, color=colors)
    axes[1].set_ylim(min(0.0, min(correlation) - 0.05), 1.0)
    axes[1].set_ylabel("Spatial correlation")
    axes[1].set_title("Top-field similarity")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "model_sensitivity_summary.png", dpi=180)
    plt.close(fig)


def write_readme(rows: list[dict], cross_obstacle_rows: list[dict]) -> None:
    model_names = {
        "Plate": "平板",
        "Sphere": "球体",
        "Hand": "手部",
    }
    relative_differences = [
        100.0 * row["relative_l2_difference"]
        for row in rows
    ]
    correlations = [
        row["spatial_correlation"]
        for row in rows
    ]
    cross_lookup = {
        (row["base"], row["compare"]): row
        for row in cross_obstacle_rows
    }
    model_order = ["Plate", "Sphere", "Hand"]
    lines = [
        "# 障碍物横向轻移实验",
        "",
        "本实验用于观察障碍物发生小幅平移时，场景顶部振幅场是否呈现连续、",
        "可用于梯度优化的响应。",
        "",
        "## 实验条件",
        "",
        "- 81 个换能器均使用硬件标定幅值，所有相位固定为 `0 rad`。",
        "- 不设置目标焦点，不进行相位聚焦。",
        "- 原始障碍物中心：`x = 48.5 mm`。",
        "- 移动后障碍物中心：`x = 51.5 mm`。",
        "- 横向位移：沿 `+x` 方向移动 `3.0 mm`。",
        "- 顶部振幅场：`z = 100 mm` 网格平面上的声压振幅。",
        f"- 计算区域：`100 x 100 x 100 mm`，网格：`{GRID_N}^3`，"
        f"步长：`{GRID_SPACING_M*1000:.6f} mm`。",
        "- 障碍物材料：密度 `1250 kg/m^3`，声速 `2200 m/s`。",
        "",
        "## 数值汇总",
        "",
        "| 模型 | 尺寸 (mm) | 左场峰值 (Pa) | 右场峰值 (Pa) | 相对 L2 差异 | 空间相关系数 | 平均绝对差 (Pa) | 最大绝对差 (Pa) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {model_names[row['model']]} | {row['dimensions_mm']} | "
            f"{row['top_peak_left_pa']:.2f} | "
            f"{row['top_peak_right_pa']:.2f} | "
            f"{100*row['relative_l2_difference']:.4f}% | "
            f"{row['spatial_correlation']:.6f} | "
            f"{row['mean_absolute_difference_pa']:.2f} | "
            f"{row['max_absolute_difference_pa']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 总览",
            "",
            "### 顶部振幅场",
            "",
            "每一行对应一种障碍物；三列依次为原始位置、移动后位置和有符号差值",
            "`右场 - 左场`。振幅图使用统一色标，差值图使用以零为中心的对称色标。",
            "",
            "![三种障碍物的顶部振幅场对比](figures/all_top_field_comparisons.png)",
            "",
            "### 跨障碍物顶部场差异",
            "",
            "这里固定使用原始位置 `x = 48.5 mm` 的顶部振幅场，比较不同障碍物",
            "本身导致的声场变化。理论上的 `3 x 3` 交叉矩阵中，对角线为零，",
            "上下三角互为相反数，因此只保留三个本质差值图：",
            "",
            "![跨障碍物顶部振幅场差异](figures/cross_obstacle_top_field_differences.png)",
            "",
            "| 行 - 列 | 平板 | 球体 | 手部 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row_name in model_order:
        values = []
        for column_name in model_order:
            if row_name == column_name:
                values.append("0")
            elif (column_name, row_name) in cross_lookup:
                pair = cross_lookup[(column_name, row_name)]
                values.append(
                    f"{100.0 * pair['symmetric_relative_l2']:.2f}% / "
                    f"corr {pair['spatial_correlation']:.3f}"
                )
            else:
                pair = cross_lookup[(row_name, column_name)]
                values.append(
                    f"{100.0 * pair['symmetric_relative_l2']:.2f}% / "
                    f"corr {pair['spatial_correlation']:.3f}"
                )
        lines.append(
            f"| {model_names[row_name]} | {values[0]} | {values[1]} | {values[2]} |"
        )
    lines.extend(
        [
            "",
            "表格中的百分比为对称归一化 L2 差异：",
            "`||行场 - 列场|| / sqrt(||行场|| ||列场||)`。",
            "三张图保留有符号的直接相减结果，标题即为差值方向。",
            "",
            "### 三维 SDF 位移叠加",
            "",
            "蓝色半透明表面为原始 SDF，红色半透明表面为沿 `+x` 移动 `3.0 mm`",
            "后的 SDF。该图按实验配置对 mesh 做平移、旋转和缩放后绘制",
            "`SDF = 0` 表面；底面绘制真实的 `9 x 9` 同相位换能器阵列，",
            "并显示 `0-100 mm` 三维坐标轴。图中不绘制三维振幅场。",
            "",
            "![三种障碍物的三维 SDF 位移叠加](figures/all_sdf_3d_comparisons.png)",
            "",
            "### 灵敏度指标",
            "",
            "![顶部场位移灵敏度汇总](figures/model_sensitivity_summary.png)",
            "",
            "## 分样例结果",
            "",
            "### 平板",
            "",
            "![平板顶部振幅场](figures/plate_top_comparison.png)",
            "",
            "![平板三维 SDF 位移叠加](figures/plate_sdf_3d_comparison.png)",
            "",
            "### 球体",
            "",
            "![球体顶部振幅场](figures/sphere_top_comparison.png)",
            "",
            "![球体三维 SDF 位移叠加](figures/sphere_sdf_3d_comparison.png)",
            "",
            "### 手部",
            "",
            "![手部顶部振幅场](figures/hand_top_comparison.png)",
            "",
            "![手部三维 SDF 位移叠加](figures/hand_sdf_3d_comparison.png)",
            "",
            "## 结果解释",
            "",
            "三种几何发生 `3.0 mm` 位移后，顶部振幅场的相对 L2 差异为",
            f"`{min(relative_differences):.2f}%` 至 `{max(relative_differences):.2f}%`，"
            f"空间相关系数为 `{min(correlations):.4f}` 至 `{max(correlations):.4f}`。",
            "这说明顶部场整体结构保持连续，同时对障碍物位置变化具有明显响应。",
            "",
            "有符号差值场在空间上呈连续分布，并保留了位移引起的增减方向。",
            "当前结果支持局部连续性，但两个采样位置不足以严格证明可微性。",
            "后续需要使用递减位移步长进行中心差分，并检查差商是否收敛。",
            "",
            "## 数据文件",
            "",
            "- `results/`：六个完整三维复声场结果。",
            "- `top_fields/`：顶部振幅、符号差值和绝对差值的 NPZ/CSV 数据。",
            "- `figures/`：本文引用的所有可视化图片。",
            "- `configs/`：六组实验配置。",
            "- `logs/`：对应求解日志。",
        ]
    )
    (STUDY / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_directories()
    create_models()
    jobs = write_configs()
    run_jobs(jobs)
    analyze()
    print(f"Study complete: {STUDY}")


if __name__ == "__main__":
    main()
