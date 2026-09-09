"""Stable-Baselines3 SAC adapter."""


def solve(problem, options):
    from baselines.rl import solve_rl
    return solve_rl(problem, options, "sac")
