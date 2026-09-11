"""Simultaneous perturbation stochastic approximation."""

from algorithm.baselines.common import (
    positive_float,
    BudgetExhausted,
    TimeBudgetExhausted,
)


def solve(problem, options):
    learning_rate = positive_float(options.get("learning_rate", 0.05), "learning_rate")
    perturbation = positive_float(options.get("perturbation", 0.1), "perturbation")
    alpha = positive_float(options.get("alpha", 0.602), "alpha")
    gamma = positive_float(options.get("gamma", 0.101), "gamma")
    current = problem.evaluate(problem.initial_phases)
    try:
        for step in range(1, problem.iterations + 1):
            delta = problem.rng.choice([-1.0, 1.0], problem.size)
            c = perturbation / step**gamma
            plus = problem.evaluate(current.phases + c * delta)
            minus = problem.evaluate(current.phases - c * delta)
            estimate = (plus.loss - minus.loss) * delta / (2 * c)
            current = problem.evaluate(current.phases - learning_rate / step**alpha * estimate)
    except BudgetExhausted as exc:
        reason = "time_budget" if isinstance(exc, TimeBudgetExhausted) else "evaluation_budget"
        return problem.result(current, reason)
    return problem.result(current, "iteration_limit")
