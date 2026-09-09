"""CMA-ES adapter using the optional cma package."""

from baselines.common import (
    BudgetExhausted, TimeBudgetExhausted, positive_float, positive_int,
)


def solve(problem, options):
    import cma

    sigma = positive_float(options.get("sigma", 0.5), "sigma")
    population = positive_int(options.get("population_size", 16), "population_size", 2)
    current = problem.evaluate(problem.initial_phases)
    strategy = cma.CMAEvolutionStrategy(current.phases, sigma, {
        "popsize": population, "seed": int(problem.cfg.training.seed) + 1,
        "verbose": -9, "verb_log": 0,
    })
    try:
        for _ in range(problem.iterations):
            if strategy.stop():
                return problem.result(problem.best, "optimizer_stopped")
            candidates = strategy.ask()
            losses = [problem.evaluate(candidate).loss for candidate in candidates]
            strategy.tell(candidates, losses)
    except BudgetExhausted as exc:
        reason = "time_budget" if isinstance(exc, TimeBudgetExhausted) else "evaluation_budget"
        return problem.result(problem.best, reason)
    return problem.result(problem.best, "iteration_limit")
