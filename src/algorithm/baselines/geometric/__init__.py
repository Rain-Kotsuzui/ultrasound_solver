"""Free-space geometric phase baseline, evaluated in the actual scene."""


def solve(problem, options):
    phases = problem.basis.solver.transducers.compute_geometric_phases()
    final = problem.evaluate(phases)
    return problem.result(final, "analytic")
