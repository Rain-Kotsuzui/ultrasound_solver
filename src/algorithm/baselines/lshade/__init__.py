"""L-SHADE-style differential evolution for black-box phase optimization."""

import math

import numpy as np

from algorithm.baselines.common import (
    BudgetExhausted,
    TimeBudgetExhausted,
    positive_float,
    positive_int,
    wrap,
)


def _phase_delta(left, right):
    """Shortest signed angular difference left - right."""
    return np.angle(np.exp(1j * (left - right)))


def _sample_scale(rng, mean):
    for _ in range(32):
        value = mean + 0.1 * rng.standard_cauchy()
        if value > 0:
            return min(value, 1.0)
    return 1.0


def _sample_crossover(rng, mean):
    return float(np.clip(rng.normal(mean, 0.1), 0.0, 1.0))


def _memory_update(memory_f, memory_cr, slot, successful_f, successful_cr, gains):
    if not successful_f:
        return slot
    weights = np.asarray(gains, dtype=np.float64)
    weights /= np.sum(weights)
    factors = np.asarray(successful_f, dtype=np.float64)
    crossovers = np.asarray(successful_cr, dtype=np.float64)
    memory_f[slot] = np.sum(weights * factors**2) / np.sum(weights * factors)
    memory_cr[slot] = np.sum(weights * crossovers)
    return (slot + 1) % len(memory_f)


def solve(problem, options):
    """Run success-history adaptive DE without accessing gradients or basis rows."""
    population_size = positive_int(options.get("population_size", 32), "population_size", 4)
    min_population_size = positive_int(
        options.get("min_population_size", 8), "min_population_size", 4
    )
    memory_size = positive_int(options.get("memory_size", 6), "memory_size")
    p_best_rate = positive_float(options.get("p_best_rate", 0.15), "p_best_rate")
    convergence_patience = positive_int(
        options.get("convergence_patience", 25), "convergence_patience"
    )
    convergence_relative_tolerance = positive_float(
        options.get("convergence_relative_tolerance", 2.0e-4),
        "convergence_relative_tolerance",
    )
    if p_best_rate > 1.0:
        raise ValueError("p_best_rate must be <= 1")
    if min_population_size > population_size:
        raise ValueError("min_population_size must not exceed population_size")
    population_size = min(population_size, problem.evaluation_budget)
    if population_size < 4:
        raise ValueError("max_evaluations must allow an L-SHADE population of four")

    current = problem.evaluate(problem.initial_phases)
    values = [current]
    try:
        while len(values) < population_size:
            values.append(problem.evaluate(problem.rng.uniform(0, 2 * np.pi, problem.size)))

        phases = np.asarray([value.phases for value in values])
        losses = np.asarray([value.loss for value in values])
        memory_f = np.full(memory_size, 0.5)
        memory_cr = np.full(memory_size, 0.5)
        memory_slot = 0
        archive = []
        checkpoint_loss = problem.best.loss
        generations_since_checkpoint = 0
        for _ in range(problem.iterations):
            count = len(values)
            order = np.argsort(losses)
            p_count = max(2, int(math.ceil(p_best_rate * count)))
            next_values = list(values)
            successful_f, successful_cr, gains = [], [], []
            accepted_originals = []

            for index in range(count):
                memory_index = int(problem.rng.integers(memory_size))
                factor = _sample_scale(problem.rng, memory_f[memory_index])
                crossover = _sample_crossover(problem.rng, memory_cr[memory_index])
                pbest = int(problem.rng.choice(order[:p_count]))
                available = [item for item in range(count) if item != index]
                r1 = int(problem.rng.choice(available))
                r2_candidates = [
                    phases[item]
                    for item in range(count)
                    if item not in {index, r1}
                ]
                r2_candidates.extend(archive)
                r2 = r2_candidates[int(problem.rng.integers(len(r2_candidates)))]
                donor = (
                    phases[index]
                    + factor * _phase_delta(phases[pbest], phases[index])
                    + factor * _phase_delta(phases[r1], r2)
                )
                mask = problem.rng.random(problem.size) < crossover
                mask[int(problem.rng.integers(problem.size))] = True
                trial_phases = wrap(np.where(mask, donor, phases[index]))
                trial = problem.evaluate(trial_phases)
                if trial.loss <= values[index].loss:
                    next_values[index] = trial
                    gain = values[index].loss - trial.loss
                    # Neutral replacements are useful for population diversity, but
                    # cannot define success-history weights.
                    if gain > 0:
                        successful_f.append(factor)
                        successful_cr.append(crossover)
                        gains.append(gain)
                    accepted_originals.append(phases[index].copy())

            values = next_values
            phases = np.asarray([value.phases for value in values])
            losses = np.asarray([value.loss for value in values])
            archive.extend(accepted_originals)
            archive_limit = max(1, population_size)
            if len(archive) > archive_limit:
                selected = problem.rng.choice(len(archive), archive_limit, replace=False)
                archive = [archive[item] for item in selected]
            memory_slot = _memory_update(
                memory_f, memory_cr, memory_slot, successful_f, successful_cr, gains
            )

            progress = problem.field_evaluations / problem.evaluation_budget
            desired = max(
                min_population_size,
                int(round(population_size - (population_size - min_population_size) * progress)),
            )
            if desired < len(values):
                retained = np.argsort(losses)[:desired]
                values = [values[item] for item in retained]
                phases = np.asarray([value.phases for value in values])
                losses = np.asarray([value.loss for value in values])

            generations_since_checkpoint += 1
            if generations_since_checkpoint >= convergence_patience:
                relative_improvement = (
                    checkpoint_loss - problem.best.loss
                ) / max(1.0, abs(checkpoint_loss))
                if relative_improvement <= convergence_relative_tolerance:
                    return problem.result(
                        problem.best,
                        "converged",
                        convergence_patience=convergence_patience,
                        convergence_relative_tolerance=convergence_relative_tolerance,
                        convergence_relative_improvement=relative_improvement,
                    )
                checkpoint_loss = problem.best.loss
                generations_since_checkpoint = 0

            if not successful_f and problem.remaining < len(values):
                return problem.result(min(values, key=lambda value: value.loss), "no_improvement")
    except BudgetExhausted as exc:
        reason = "time_budget" if isinstance(exc, TimeBudgetExhausted) else "evaluation_budget"
        return problem.result(problem.best, reason)

    return problem.result(min(values, key=lambda value: value.loss), "iteration_limit")
