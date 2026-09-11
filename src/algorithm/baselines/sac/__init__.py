"""Stable-Baselines3 SAC adapter."""


def solve(problem, options):
    from algorithm.baselines.rl import solve_rl
    return solve_rl(problem, options, "sac")
