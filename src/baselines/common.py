"""Shared objective, accounting and results for phase algorithms."""

from dataclasses import dataclass, field
import time

import numpy as np

from training.phase_adjoint_optimizer import (
    AmplitudeTarget, AmplitudeFieldLoss, FocalPressureLoss, FocalContrastLoss,
    _build_background_mask, _nearest_grid_index, _validate_target_point,
)


class BudgetExhausted(RuntimeError):
    pass


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def positive_float(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def wrap(phases):
    phases = np.asarray(phases, dtype=np.float64)
    if not np.isfinite(phases).all():
        raise ValueError("phases must be finite")
    return np.mod(phases, 2 * np.pi)


@dataclass
class Evaluation:
    phases: np.ndarray
    loss: float
    field: np.ndarray
    metrics: dict
    gradient: np.ndarray | None = None


@dataclass
class AlgorithmResult:
    final: Evaluation
    best: Evaluation
    history: list
    termination_reason: str
    metadata: dict = field(default_factory=dict)


class PhaseProblem:
    def __init__(self, cfg, basis, loss_curve=None):
        self.cfg, self.basis = cfg, basis
        self.size = basis.num_transducers
        self.target = AmplitudeTarget.from_config(cfg)
        for point in cfg.targets:
            _validate_target_point(cfg, np.asarray(point))
        if cfg.training.loss_type != "field_match" and not cfg.targets:
            raise ValueError("Focal losses require targets")
        factories = {
            "field_match": lambda: AmplitudeFieldLoss(self.target),
            "focal_pressure": lambda: FocalPressureLoss(cfg, self.target),
            "focal_contrast": lambda: FocalContrastLoss(cfg, self.target),
        }
        if cfg.training.loss_type not in factories:
            raise ValueError("Unknown training.loss_type")
        self.loss_model = factories[cfg.training.loss_type]()
        if (not np.isfinite(self.target.target).all()
                or not np.isfinite(self.target.weight).all()
                or np.any(self.target.target < 0) or np.any(self.target.weight < 0)):
            raise ValueError("Target amplitudes and weights must be finite and nonnegative")
        self.target_indices = [_nearest_grid_index(cfg, np.asarray(p)) for p in cfg.targets]
        self.background_mask = _build_background_mask(
            cfg, np.asarray(cfg.targets), cfg.training.target_radius,
            cfg.training.source_exclusion_layers,
        )
        if not np.any(self.background_mask):
            raise ValueError("Evaluation background mask is empty")
        self.evaluation_budget = positive_int(
            cfg.training.max_evaluations, "max_evaluations"
        )
        self._reserved_evaluations = 0
        self.iterations = positive_int(cfg.training.iterations, "iterations", 0)
        self.rng = np.random.default_rng(cfg.training.seed)
        self.initial_phases = self._initial_phases()
        self.field_evaluations = 0
        self.vjp_evaluations = 0
        self.history = []
        self.best = None
        self.last = None
        self.started = time.perf_counter()
        if loss_curve is None:
            from baselines.loss_curve import LiveLossCurve

            loss_curve = LiveLossCurve(
                cfg.training.show_loss_curve,
                cfg.training.loss_curve_update_interval,
                cfg.training.loss_curve_pause_seconds,
                cfg.algorithm,
            )
        self.loss_curve = loss_curve

    def _initial_phases(self):
        mode = self.cfg.training.initial_phase
        if mode == "zero":
            return np.zeros(self.size)
        if mode == "current":
            return wrap(self.basis.solver.transducers.phases)
        if mode == "random":
            return self.rng.uniform(0, 2 * np.pi, self.size)
        if mode == "geometric":
            return wrap(self.basis.solver.transducers.compute_geometric_phases())
        if mode == "response":
            from baselines.response_alignment import phases_for_targets
            return phases_for_targets(self)
        raise ValueError("initial_phase must be geometric, response, zero, current or random")

    @property
    def remaining(self):
        return self.evaluation_budget - self._reserved_evaluations - self.field_evaluations

    def metrics(self, field):
        amplitude = np.abs(field)
        values = np.array([amplitude[i] for i in self.target_indices])
        if not values.size:
            values = amplitude[self.target.weight >= np.max(self.target.weight) * 0.5]
        mean, peak = float(np.mean(values)), float(np.max(values))
        background = float(np.max(amplitude[self.background_mask]))
        norm = float(np.linalg.norm(self.target.target))
        error = float(np.linalg.norm(amplitude - self.target.target))
        return {
            "target_mean": mean,
            "target_min": float(np.min(values)),
            "background_max": background,
            "contrast": mean / background if background > 0 else None,
            "uniformity": float(np.min(values)) / peak if peak > 0 else None,
            "target_cv": float(np.std(values)) / mean if mean > 0 else None,
            "field_relative_l2": error / norm if norm > 0 else None,
            "field_absolute_l2": error,
        }

    def evaluate(self, phases, gradient=False, stage="optimization"):
        if self.remaining <= 0:
            raise BudgetExhausted("field evaluation budget exhausted")
        phases = wrap(phases)
        if phases.shape != (self.size,):
            raise ValueError(f"Expected {self.size} phases")
        self.field_evaluations += 1
        # return_numpy=True synchronizes GPU field work before timing/logging.
        field_value = self.basis.field(phases, return_numpy=True)
        loss, cotangent = self.loss_model.evaluate(field_value, with_cotangent=gradient)
        if not np.isfinite(loss) or not np.isfinite(field_value).all():
            raise FloatingPointError("Non-finite field or loss")
        grad = None
        if gradient:
            self.vjp_evaluations += 1
            grad = self.basis.phase_vjp(cotangent, phases, return_numpy=True)
            if not np.isfinite(grad).all():
                raise FloatingPointError("Non-finite phase gradient")
        evaluation = Evaluation(phases.copy(), loss, field_value, self.metrics(field_value), grad)
        self.last = evaluation
        if self.best is None or loss < self.best.loss:
            self.best = evaluation
        row = {
            "evaluation": self.field_evaluations, "vjp_evaluations": self.vjp_evaluations,
            "elapsed_seconds": time.perf_counter() - self.started,
            "loss": loss, "best_loss": self.best.loss, "stage": stage,
            "gradient_norm": float(np.linalg.norm(grad)) if grad is not None else None,
            **evaluation.metrics,
        }
        self.history.append(row)
        self.loss_curve.update(row)
        print(f"[{self.cfg.algorithm}] eval={self.field_evaluations:05d} "
              f"stage={stage} loss={loss:.6e} best={self.best.loss:.6e} "
              f"target={row['target_mean']:.3f} background={row['background_max']:.3f}")
        return evaluation

    def reserve_evaluations(self, count):
        count = positive_int(count, "reserved evaluation count", 0)
        if self.remaining < count:
            raise ValueError(
                f"max_evaluations={self.evaluation_budget} cannot reserve {count} "
                "evaluations after work already completed"
            )
        self._reserved_evaluations += count

    def release_evaluations(self, count):
        count = positive_int(count, "reserved evaluation count", 0)
        if count > self._reserved_evaluations:
            raise ValueError("Cannot release more evaluations than reserved")
        self._reserved_evaluations -= count

    def result(self, final, reason, **metadata):
        if final is None or self.best is None:
            raise RuntimeError("Algorithm returned no evaluated candidate")
        self.loss_curve.finalize(reason)
        return AlgorithmResult(
            final, self.best, list(self.history), reason,
            {"field_evaluations": self.field_evaluations,
             "vjp_evaluations": self.vjp_evaluations,
             "elapsed_seconds": time.perf_counter() - self.started, **metadata},
        )
