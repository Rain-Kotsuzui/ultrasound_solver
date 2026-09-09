"""Response-basis analytic VJP with L-BFGS-B or Adam."""

import numpy as np

from baselines.common import BudgetExhausted, TimeBudgetExhausted, positive_float


def solve(problem, options):
    method = options.get("optimizer", "adam")
    if method not in {"lbfgsb", "adam"}:
        raise ValueError("adjoint optimizer must be lbfgsb or adam")
    current = problem.evaluate(problem.initial_phases, gradient=True)
    checks = []
    try:
        if options.get("gradient_check", False):
            h = positive_float(options.get("gradient_check_step", 1e-3),
                               "gradient_check_step")
            for index in range(min(8, problem.size)):
                offset = np.zeros(problem.size)
                offset[index] = h
                plus = problem.evaluate(current.phases + offset, stage="gradient_check")
                minus = problem.evaluate(current.phases - offset, stage="gradient_check")
                numerical = (plus.loss - minus.loss) / (2 * h)
                exact = float(current.gradient[index])
                error = abs(exact - numerical) / max(abs(exact), abs(numerical), 1.0)
                checks.append({"index": index, "analytic": exact,
                               "finite_difference": numerical, "scaled_error": error})
                print(f"[GradCheck] emitter={index} analytic={exact:.6e} "
                      f"finite_diff={numerical:.6e} scaled_error={error:.3e}")
        if problem.iterations == 0:
            return problem.result(current, "iteration_limit", gradient_checks=checks)
        if method == "adam":
            rate = positive_float(options.get("learning_rate", 0.05),
                                  "learning_rate")
            first = np.zeros(problem.size)
            second = np.zeros(problem.size)
            for step in range(1, problem.iterations + 1):
                first = 0.9 * first + 0.1 * current.gradient
                second = 0.999 * second + 0.001 * current.gradient**2
                update = rate * (first / (1 - 0.9**step)) / (
                    np.sqrt(second / (1 - 0.999**step)) + 1e-8)
                current = problem.evaluate(current.phases - update, gradient=True)
            return problem.result(current, "iteration_limit", gradient_checks=checks)

        from scipy.optimize import minimize

        def objective(phases):
            nonlocal current
            # Reuse the initial evaluation and repeated line-search requests.
            if not np.array_equal(phases, current.phases):
                current = problem.evaluate(phases, gradient=True)
            return current.loss, current.gradient

        result = minimize(
            objective, current.phases, jac=True, method="L-BFGS-B",
            bounds=[(0, 2 * np.pi)] * problem.size,
            options={"maxiter": problem.iterations, "ftol": 1e-12,
                     "gtol": 1e-8, "maxls": 40},
        )
        if not np.array_equal(np.mod(result.x, 2 * np.pi), current.phases):
            current = problem.evaluate(result.x, gradient=True)
        return problem.result(current, "converged" if result.success else "optimizer_stopped",
                              optimizer_message=str(result.message), gradient_checks=checks)
    except BudgetExhausted as exc:
        reason = "time_budget" if isinstance(exc, TimeBudgetExhausted) else "evaluation_budget"
        return problem.result(current, reason, gradient_checks=checks)
