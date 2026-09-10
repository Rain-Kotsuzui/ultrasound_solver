"""Shared objective, accounting and results for phase algorithms."""

from dataclasses import dataclass, field
import time

import cupy as cp
import numpy as np

from training.phase_adjoint_optimizer import (
    AmplitudeTarget, AmplitudeFieldLoss, FocalPressureLoss, FocalContrastLoss,
    _build_background_mask, _nearest_grid_index, _validate_target_point,
)


class BudgetExhausted(RuntimeError):
    pass


class TimeBudgetExhausted(BudgetExhausted):
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
        self.time_budget_seconds = float(cfg.training.max_seconds)
        if (not np.isfinite(self.time_budget_seconds)
                or self.time_budget_seconds < 0):
            raise ValueError("max_seconds must be finite and nonnegative")
        self._reserved_evaluations = 0
        self.iterations = positive_int(cfg.training.iterations, "iterations", 0)
        self.rng = np.random.default_rng(cfg.training.seed)
        self.initial_phases = self._initial_phases()
        self.field_evaluations = 0
        self.vjp_evaluations = 0
        self.history = []
        self.best = None
        self.last = None
        if loss_curve is None:
            from baselines.loss_curve import LiveLossCurve

            loss_curve = LiveLossCurve(
                cfg.training.show_loss_curve,
                cfg.training.loss_curve_update_interval,
                cfg.training.loss_curve_pause_seconds,
                cfg.algorithm,
            )
        self.loss_curve = loss_curve
        if (
            getattr(self.basis, "basis_gpu", None) is not None
            and cfg.algorithm == "adjoint"
        ):
            self._warm_gpu_pipeline()
        self.started = time.perf_counter()

    def _warm_gpu_pipeline(self):
        field = self.basis.field(self.initial_phases, return_numpy=False)
        loss, cotangent = self.loss_model.evaluate_device(field)
        self.basis.phase_vjp(cotangent, self.initial_phases, return_numpy=False)
        self.metrics_device(field)
        if not bool(cp.isfinite(loss)):
            raise FloatingPointError("Non-finite GPU warm-up loss")
        cp.cuda.Stream.null.synchronize()

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

    @property
    def elapsed_seconds(self):
        return time.perf_counter() - self.started

    @property
    def time_remaining_seconds(self):
        if self.time_budget_seconds == 0:
            return float("inf")
        return self.time_budget_seconds - self.elapsed_seconds

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

    def metrics_device(self, field):
        amplitude = cp.abs(field)
        if self.target_indices:
            indices = tuple(np.asarray(self.target_indices, dtype=np.intp).T)
            values = amplitude[indices]
        else:
            target_mask = self._device_target_mask()
            values = amplitude[target_mask]
        background_mask = self._device_background_mask()
        target = self._device_target()
        mean = cp.mean(values)
        peak = cp.max(values)
        background = cp.max(amplitude[background_mask])
        error = cp.linalg.norm(amplitude - target)
        target_min = cp.min(values)
        target_std = cp.std(values)
        summary = cp.asnumpy(
            cp.asarray(
                [
                    mean, target_min, background, peak, target_std, error,
                ]
            )
        )
        mean, target_min, background, peak, target_std, error = (
            float(value) for value in summary
        )
        norm = float(np.linalg.norm(self.target.target))
        return {
            "target_mean": mean,
            "target_min": target_min,
            "background_max": background,
            "contrast": mean / background if background > 0 else None,
            "uniformity": target_min / peak if peak > 0 else None,
            "target_cv": target_std / mean if mean > 0 else None,
            "field_relative_l2": error / norm if norm > 0 else None,
            "field_absolute_l2": error,
        }

    def _device_target(self):
        if not hasattr(self, "_gpu_target"):
            self._gpu_target = cp.asarray(self.target.target)
        return self._gpu_target

    def _device_target_mask(self):
        if not hasattr(self, "_gpu_target_mask"):
            self._gpu_target_mask = cp.asarray(
                self.target.weight >= np.max(self.target.weight) * 0.5
            )
        return self._gpu_target_mask

    def _device_background_mask(self):
        if not hasattr(self, "_gpu_background_mask"):
            self._gpu_background_mask = cp.asarray(self.background_mask)
        return self._gpu_background_mask

    def evaluate(self, phases, gradient=False, stage="optimization"):
        # Every method must produce one valid initial candidate for reporting.
        if self.field_evaluations > 0 and self.time_remaining_seconds <= 0:
            raise TimeBudgetExhausted("wall-clock time budget exhausted")
        if self.remaining <= 0:
            raise BudgetExhausted("field evaluation budget exhausted")
        phases = wrap(phases)
        if phases.shape != (self.size,):
            raise ValueError(f"Expected {self.size} phases")
        self.field_evaluations += 1
        use_gpu = getattr(self.basis, "basis_gpu", None) is not None
        field_value = self.basis.field(phases, return_numpy=not use_gpu)
        if use_gpu:
            loss_value, cotangent = self.loss_model.evaluate_device(
                field_value,
                with_cotangent=gradient,
            )
            loss = float(loss_value)
        else:
            loss, cotangent = self.loss_model.evaluate(
                field_value,
                with_cotangent=gradient,
            )
        if not np.isfinite(loss) or (
            use_gpu and not bool(cp.isfinite(field_value).all())
        ):
            raise FloatingPointError("Non-finite field or loss")
        grad = None
        if gradient:
            self.vjp_evaluations += 1
            grad = self.basis.phase_vjp(
                cotangent,
                phases,
                return_numpy=True,
            )
            if not np.isfinite(grad).all():
                raise FloatingPointError("Non-finite phase gradient")
        metrics = self.metrics_device(field_value) if use_gpu else self.metrics(field_value)
        evaluation = Evaluation(
            phases.copy(),
            loss,
            None if use_gpu else field_value,
            metrics,
            grad,
        )
        self.last = evaluation
        if self.best is None or loss < self.best.loss:
            self.best = evaluation
        row = {
            "evaluation": self.field_evaluations, "vjp_evaluations": self.vjp_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
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
        if final.field is None:
            final.field = self.basis.field(final.phases, return_numpy=True)
        if self.best.field is None:
            self.best.field = self.basis.field(self.best.phases, return_numpy=True)
        self.loss_curve.finalize(reason)
        return AlgorithmResult(
            final, self.best, list(self.history), reason,
            {"field_evaluations": self.field_evaluations,
             "vjp_evaluations": self.vjp_evaluations,
             "elapsed_seconds": self.elapsed_seconds,
             "time_budget_seconds": self.time_budget_seconds, **metadata},
        )
