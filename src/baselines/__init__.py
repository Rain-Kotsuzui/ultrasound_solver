"""Algorithm selection without importing optional RL dependencies."""

from importlib import import_module, util

ALGORITHMS = (
    "adjoint", "geometric", "response_alignment", "gabs", "spsa", "sac", "ppo", "cmaes",
)
OPTIONS = {
    "adjoint": {"optimizer", "learning_rate", "gradient_check", "gradient_check_step"},
    "geometric": set(),
    "response_alignment": set(),
    "gabs": {"phase_levels"},
    "spsa": {"learning_rate", "perturbation", "alpha", "gamma"},
    "cmaes": {"sigma", "population_size"},
    "sac": {"episode_steps", "evaluation_steps", "action_scale", "reward_scale",
            "total_timesteps", "run_mode", "checkpoint", "random_reset", "device",
            "learning_rate", "batch_size", "learning_starts", "buffer_size"},
    "ppo": {"episode_steps", "evaluation_steps", "action_scale", "reward_scale",
            "total_timesteps", "run_mode", "checkpoint", "random_reset", "device",
            "learning_rate", "batch_size", "n_steps"},
}


def validate_algorithm(cfg):
    if cfg.algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
    unknown = set(cfg.algorithm_options) - OPTIONS[cfg.algorithm]
    if unknown:
        raise ValueError(f"Unsupported {cfg.algorithm} options: {sorted(unknown)}")
    dependencies = {"sac": ("stable_baselines3", "gymnasium"),
                    "ppo": ("stable_baselines3", "gymnasium"), "cmaes": ("cma",)}
    missing = [name for name in dependencies.get(cfg.algorithm, ())
               if util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Missing {', '.join(missing)}. "
                          "Install: python -m pip install -r src/baselines/requirements.txt")


def run(cfg, basis):
    from baselines.common import PhaseProblem

    validate_algorithm(cfg)
    problem = PhaseProblem(cfg, basis)
    module = import_module(f"baselines.{cfg.algorithm}")
    return problem, module.solve(problem, cfg.algorithm_options)
