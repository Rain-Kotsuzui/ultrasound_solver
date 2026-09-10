"""Run the published algorithm-comparison scenario from the repository root."""

from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from compare import main


ALGORITHMS = (
    "geometric",
    "response_alignment",
    "adjoint",
    "gabs",
    "spsa",
    "cmaes",
    "lshade",
    "sac",
    "ppo",
)


if __name__ == "__main__":
    defaults = [
        "--config",
        str(
            ROOT / "algorithm_comparison" / "config"
            / "phase_oblique_reflecting_x_12x12.yaml"
        ),
        "--output-dir",
        str(ROOT / "algorithm_comparison" / "results"),
        "--algorithms",
        ",".join(ALGORITHMS),
    ]
    if "--config" not in sys.argv:
        sys.argv.extend(defaults[:2])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(defaults[2:4])
    if "--algorithms" not in sys.argv:
        sys.argv.extend(defaults[4:])
    main()
