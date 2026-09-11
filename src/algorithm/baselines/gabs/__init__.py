"""GABS-style greedy coordinate search, not a claimed paper reproduction."""

import numpy as np

from algorithm.baselines.common import (
    BudgetExhausted,
    TimeBudgetExhausted,
    positive_int,
)


def solve(problem, options):
    levels = positive_int(options.get("phase_levels", 16), "phase_levels", 2)
    candidates = np.linspace(0, 2 * np.pi, levels, endpoint=False)
    current = problem.evaluate(problem.initial_phases)
    try:
        for _ in range(problem.iterations):
            improved = False
            for index in range(problem.size):
                selected = current
                for phase in candidates:
                    phases = current.phases.copy()
                    phases[index] = phase
                    value = problem.evaluate(phases)
                    if value.loss < selected.loss:
                        selected = value
                improved |= selected.loss < current.loss
                current = selected
            if not improved:
                return problem.result(current, "no_improvement")
    except BudgetExhausted as exc:
        # The interrupted coordinate may contain a better evaluated candidate.
        reason = "time_budget" if isinstance(exc, TimeBudgetExhausted) else "evaluation_budget"
        return problem.result(problem.best, reason)
    return problem.result(current, "iteration_limit")
