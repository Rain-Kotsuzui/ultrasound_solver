from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PhaseOptimizationState:
    iteration: int
    loss: float
    gradient_norm: float
    target_mean: float
    background_mean: float


class AmplitudeTarget:
    def __init__(self, target: np.ndarray, weight: np.ndarray):
        if target.shape != weight.shape:
            raise ValueError("target and weight must have the same shape")
        self.target = np.asarray(target, dtype=np.float64)
        self.weight = np.asarray(weight, dtype=np.float64)

    @classmethod
    def from_config(cls, cfg):
        training = cfg.training
        if training.target_field_file:
            return cls._from_file(cfg, Path(training.target_field_file))
        return cls._from_targets(cfg)

    @classmethod
    def _from_file(cls, cfg, path: Path):
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.suffix.lower() == ".npy":
            target = np.load(path)
            weight = np.ones_like(target, dtype=np.float64)
        elif path.suffix.lower() == ".npz":
            data = np.load(path)
            target = np.asarray(data["target"], dtype=np.float64)
            if "weight" in data:
                weight = np.asarray(data["weight"], dtype=np.float64)
            else:
                weight = np.ones_like(target, dtype=np.float64)
            data.close()
        else:
            raise ValueError("target_field_file must be .npy or .npz")
        cls._validate_shape(cfg, target)
        return cls(target, weight)

    @classmethod
    def _from_targets(cls, cfg):
        if not cfg.targets:
            raise ValueError(
                "phase_only optimization requires targets or "
                "training.target_field_file"
            )
        target, weight = build_gaussian_target_fields(
            cfg=cfg,
            target_points=np.asarray(cfg.targets, dtype=np.float64),
            peak_pressure=float(cfg.training.target_peak_pressure),
            sigma=float(cfg.training.target_sigma),
            background_pressure=float(cfg.training.background_pressure),
            target_weight=float(cfg.training.target_weight),
            background_weight=float(cfg.training.background_weight),
        )
        return cls(target, weight)

    @staticmethod
    def _validate_shape(cfg, target: np.ndarray):
        expected = (
            cfg.domain.nx,
            cfg.domain.ny,
            cfg.domain.nz,
        )
        if target.shape != expected:
            raise ValueError(
                f"target field shape must be {expected}, got {target.shape}"
            )


def build_gaussian_target_fields(
    cfg,
    target_points: np.ndarray,
    peak_pressure: float,
    sigma: float,
    background_pressure: float = 0.0,
    target_weight: float = 1.0,
    background_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build full-resolution target amplitude and weight fields.

    Each requested target point creates one Gaussian lobe. The lobe is
    normalized at the nearest grid sample to keep all target samples at the
    same peak amplitude. Multiple lobes are combined by maximum so nearby
    targets do not raise each other's requested peak pressure.
    """
    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("target_points must have shape (num_targets, 3)")
    if points.shape[0] == 0:
        raise ValueError("at least one target point is required")
    if sigma <= 0.0:
        raise ValueError("target sigma must be positive")
    if peak_pressure < background_pressure:
        raise ValueError("peak_pressure must be >= background_pressure")
    shape = (
        cfg.domain.nx,
        cfg.domain.ny,
        cfg.domain.nz,
    )
    target = np.full(shape, float(background_pressure), dtype=np.float64)
    weight = np.full(shape, float(background_weight), dtype=np.float64)

    xs = np.linspace(0.0, cfg.domain.lx, cfg.domain.nx)
    ys = np.linspace(0.0, cfg.domain.ly, cfg.domain.ny)
    zs = np.linspace(0.0, cfg.domain.lz, cfg.domain.nz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    sigma_sq = sigma * sigma
    for point in points:
        _validate_target_point(cfg, point)
        center = np.asarray(point, dtype=np.float64)
        radius_sq = (
            (x - center[0]) ** 2
            + (y - center[1]) ** 2
            + (z - center[2]) ** 2
        )
        gaussian = np.exp(-0.5 * radius_sq / sigma_sq)
        index = _nearest_grid_index(cfg, center)
        gaussian = gaussian / gaussian[index]
        target = np.maximum(
            target,
            background_pressure
            + (peak_pressure - background_pressure) * gaussian,
        )
        weight = np.maximum(
            weight,
            background_weight
            + (target_weight - background_weight) * gaussian,
        )
    return target, weight


def _nearest_grid_index(cfg, point: np.ndarray) -> tuple[int, int, int]:
    dx = cfg.domain.dx
    return (
        int(round(point[0] / dx)),
        int(round(point[1] / dx)),
        int(round(point[2] / dx)),
    )


def _validate_target_point(cfg, point: np.ndarray) -> None:
    lower = np.zeros(3, dtype=np.float64)
    upper = np.array([cfg.domain.lx, cfg.domain.ly, cfg.domain.lz])
    if np.any(point < lower) or np.any(point > upper):
        raise ValueError(
            "target point is outside the domain: "
            f"{point.tolist()} not in [0, {upper.tolist()}]"
        )


class AmplitudeFieldLoss:
    def __init__(self, amplitude_target: AmplitudeTarget, eps: float = 1.0e-12):
        self.target = amplitude_target.target
        self.weight = amplitude_target.weight
        self.eps = float(eps)

    def evaluate(self, field: np.ndarray) -> tuple[float, np.ndarray]:
        amplitude = np.abs(field)
        residual = amplitude - self.target
        weighted = self.weight * residual
        loss = 0.5 * float(np.vdot(weighted, weighted).real)
        cotangent = (
            self.weight
            * self.weight
            * residual
            * field
            / (amplitude + self.eps)
        )
        return loss, cotangent.astype(np.complex128, copy=False)

    def stats(self, field: np.ndarray) -> tuple[float, float]:
        amplitude = np.abs(field)
        target_mask = self.weight >= 0.5 * float(np.max(self.weight))
        background_mask = ~target_mask
        target_mean = float(np.mean(amplitude[target_mask]))
        if np.any(background_mask):
            background_mean = float(np.mean(amplitude[background_mask]))
        else:
            background_mean = 0.0
        return target_mean, background_mean


class FocalPressureLoss:
    def __init__(
        self,
        cfg,
        amplitude_target: AmplitudeTarget,
        eps: float = 1.0e-12,
    ):
        self.cfg = cfg
        self.target = amplitude_target.target
        self.weight = amplitude_target.weight
        self.eps = float(eps)
        self.target_indices = [
            _nearest_grid_index(cfg, np.asarray(point, dtype=np.float64))
            for point in cfg.targets
        ]
        self.target_weight = float(cfg.training.target_weight)
        self.background_weight = float(cfg.training.background_weight)

    def evaluate(self, field: np.ndarray) -> tuple[float, np.ndarray]:
        amplitude = np.abs(field)
        cotangent = np.zeros_like(field, dtype=np.complex128)
        focal_loss = 0.0
        scale = self.target_weight / len(self.target_indices)
        for index in self.target_indices:
            value = amplitude[index]
            focal_loss -= scale * float(value)
            cotangent[index] -= scale * field[index] / (value + self.eps)

        background_loss = 0.0
        if self.background_weight > 0.0:
            background_loss = 0.5 * self.background_weight * float(
                np.mean(amplitude * amplitude)
            )
            cotangent += (
                self.background_weight
                / field.size
                * field
            )
        return focal_loss + background_loss, cotangent

    def stats(self, field: np.ndarray) -> tuple[float, float]:
        amplitude = np.abs(field)
        target_values = [
            amplitude[index]
            for index in self.target_indices
        ]
        target_mean = float(np.mean(target_values))
        background_mean = float(np.mean(amplitude))
        return target_mean, background_mean


class FocalContrastLoss:
    def __init__(
        self,
        cfg,
        amplitude_target: AmplitudeTarget,
        eps: float = 1.0e-12,
    ):
        self.cfg = cfg
        self.target = amplitude_target.target
        self.weight = amplitude_target.weight
        self.eps = float(eps)
        self.target_indices = [
            _nearest_grid_index(cfg, np.asarray(point, dtype=np.float64))
            for point in cfg.targets
        ]
        self.target_weight = float(cfg.training.target_weight)
        self.sidelobe_weight = float(cfg.training.background_weight)
        self.temperature = float(cfg.training.sidelobe_temperature)
        if self.temperature <= 0.0:
            raise ValueError("training.sidelobe_temperature must be positive")
        self.background_mask = _build_background_mask(
            cfg,
            np.asarray(cfg.targets, dtype=np.float64),
            float(cfg.training.target_radius),
            int(cfg.training.source_exclusion_layers),
        )
        if not np.any(self.background_mask):
            raise ValueError("focal_contrast background mask is empty")

    def evaluate(self, field: np.ndarray) -> tuple[float, np.ndarray]:
        amplitude = np.abs(field)
        cotangent = np.zeros_like(field, dtype=np.complex128)

        scale = self.target_weight / len(self.target_indices)
        target_values = []
        for index in self.target_indices:
            value = amplitude[index]
            target_values.append(value)
            cotangent[index] -= scale * field[index] / (value + self.eps)
        target_mean = float(np.mean(target_values))
        loss = -self.target_weight * target_mean

        background_values = amplitude[self.background_mask]
        maximum = float(np.max(background_values))
        shifted = (background_values - maximum) / self.temperature
        weights = np.exp(shifted)
        probabilities = weights / np.sum(weights)
        smooth_max = maximum + self.temperature * (
            np.log(np.sum(weights)) - np.log(background_values.size)
        )
        loss += self.sidelobe_weight * float(smooth_max)

        background_cotangent = (
            self.sidelobe_weight
            * probabilities
            * field[self.background_mask]
            / (background_values + self.eps)
        )
        cotangent[self.background_mask] += background_cotangent
        return loss, cotangent

    def stats(self, field: np.ndarray) -> tuple[float, float]:
        amplitude = np.abs(field)
        target_values = [
            amplitude[index]
            for index in self.target_indices
        ]
        background_max = float(np.max(amplitude[self.background_mask]))
        return float(np.mean(target_values)), background_max


def _build_background_mask(
    cfg,
    target_points: np.ndarray,
    target_radius: float,
    source_exclusion_layers: int,
) -> np.ndarray:
    if target_radius <= 0.0:
        raise ValueError("training.target_radius must be positive")
    shape = (
        cfg.domain.nx,
        cfg.domain.ny,
        cfg.domain.nz,
    )
    xs = np.linspace(0.0, cfg.domain.lx, cfg.domain.nx)
    ys = np.linspace(0.0, cfg.domain.ly, cfg.domain.ny)
    zs = np.linspace(0.0, cfg.domain.lz, cfg.domain.nz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    mask = np.ones(shape, dtype=bool)
    layers = max(int(source_exclusion_layers), 0)
    if layers > 0:
        mask[:, :, :layers] = False
    radius_sq_limit = target_radius * target_radius
    for point in target_points:
        center = np.asarray(point, dtype=np.float64)
        radius_sq = (
            (x - center[0]) ** 2
            + (y - center[1]) ** 2
            + (z - center[2]) ** 2
        )
        mask[radius_sq <= radius_sq_limit] = False
    return mask


class PhaseOnlyOptimizer:
    def __init__(self, cfg, basis):
        self.cfg = cfg
        self.basis = basis
        optimizer = str(cfg.training.optimizer).lower()
        if optimizer not in {"adam", "lbfgsb"}:
            raise ValueError("phase_only supports optimizer='adam' or 'lbfgsb'")
        self.target = AmplitudeTarget.from_config(cfg)
        self.initial_phases = self._build_initial_phases()
        loss_type = str(cfg.training.loss_type).lower()
        if loss_type == "field_match":
            self.loss_model = AmplitudeFieldLoss(self.target)
        elif loss_type == "focal_pressure":
            self.loss_model = FocalPressureLoss(cfg, self.target)
        elif loss_type == "focal_contrast":
            self.loss_model = FocalContrastLoss(cfg, self.target)
        else:
            raise ValueError(
                "training.loss_type must be 'field_match', "
                "'focal_pressure', or 'focal_contrast'"
            )

    def _build_initial_phases(self) -> np.ndarray:
        initial_phase = str(self.cfg.training.initial_phase).lower()
        if initial_phase == "zero":
            return np.zeros(
                self.basis.solver.transducers.num_transducers,
                dtype=np.float64,
            )
        if initial_phase == "current":
            return np.array(
                self.basis.solver.transducers.phases,
                dtype=np.float64,
            )
        if initial_phase == "baseline":
            return np.mod(
                self.basis.solver.transducers._compute_baseline_phases(),
                2.0 * np.pi,
            )
        if initial_phase == "response":
            return self._response_focus_phases()
        raise ValueError(
            "training.initial_phase must be 'baseline', 'response', "
            "'current', or 'zero'"
        )

    def _response_focus_phases(self) -> np.ndarray:
        response = np.zeros(
            self.basis.solver.transducers.num_transducers,
            dtype=np.complex128,
        )
        basis_matrix = self.basis.basis
        shape = (
            self.cfg.domain.nx,
            self.cfg.domain.ny,
            self.cfg.domain.nz,
        )
        for point in self.cfg.targets:
            index = _nearest_grid_index(
                self.cfg,
                np.asarray(point, dtype=np.float64),
            )
            flat_index = np.ravel_multi_index(index, shape, order="F")
            response += basis_matrix[flat_index, :].conjugate()
        return np.mod(np.angle(response), 2.0 * np.pi)

    def loss_and_gradient(
        self,
        phases: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        phases = np.asarray(phases, dtype=np.float64)
        field = self.basis.field(phases, return_numpy=True)
        loss, cotangent = self.loss_model.evaluate(field)
        gradient = self.basis.phase_vjp(
            cotangent,
            phases,
            return_numpy=True,
        )
        return loss, gradient, field

    def optimize(self) -> tuple[np.ndarray, list[PhaseOptimizationState]]:
        optimizer = str(self.cfg.training.optimizer).lower()
        if optimizer == "lbfgsb":
            return self._optimize_lbfgsb()
        return self._optimize_adam()

    def _optimize_adam(self) -> tuple[np.ndarray, list[PhaseOptimizationState]]:
        phases = np.array(self.initial_phases, dtype=np.float64)
        learning_rate = float(self.cfg.training.learning_rate)
        iterations = int(self.cfg.training.iterations)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1.0e-8
        first_moment = np.zeros_like(phases)
        second_moment = np.zeros_like(phases)
        history = []
        initial_loss, _, initial_field = self.loss_and_gradient(phases)
        initial_target_mean, initial_background_mean = self.loss_model.stats(
            initial_field
        )
        print(
            "[PhaseOnly] "
            f"iter={0:04d} "
            f"loss={initial_loss:.6e} "
            f"|grad|={0.0:.6e} "
            f"target_mean={initial_target_mean:.3f} Pa "
            f"background_ref={initial_background_mean:.3f} Pa "
            f"contrast={initial_target_mean / initial_background_mean:.4f}"
        )

        for iteration in range(1, iterations + 1):
            _, gradient, _ = self.loss_and_gradient(phases)
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
            corrected_first = first_moment / (1.0 - beta1**iteration)
            corrected_second = second_moment / (1.0 - beta2**iteration)
            phases = phases - learning_rate * corrected_first / (
                np.sqrt(corrected_second) + eps
            )
            phases = np.mod(phases, 2.0 * np.pi)
            loss, _, field = self.loss_and_gradient(phases)
            target_mean, background_mean = self.loss_model.stats(field)
            state = PhaseOptimizationState(
                iteration=iteration,
                loss=loss,
                gradient_norm=float(np.linalg.norm(gradient)),
                target_mean=target_mean,
                background_mean=background_mean,
            )
            history.append(state)
            print(
                "[PhaseOnly] "
                f"iter={iteration:04d} "
                f"loss={state.loss:.6e} "
                f"|grad|={state.gradient_norm:.6e} "
                f"target_mean={state.target_mean:.3f} Pa "
                f"background_ref={state.background_mean:.3f} Pa "
                f"contrast={state.target_mean / state.background_mean:.4f}"
            )
        return phases, history

    def _optimize_lbfgsb(self) -> tuple[np.ndarray, list[PhaseOptimizationState]]:
        from scipy.optimize import minimize

        phases = np.array(self.initial_phases, dtype=np.float64)
        iterations = int(self.cfg.training.iterations)
        history = []

        initial_loss, initial_gradient, initial_field = self.loss_and_gradient(phases)
        initial_target_mean, initial_background_mean = self.loss_model.stats(
            initial_field
        )
        print(
            "[PhaseOnly] "
            f"iter={0:04d} "
            f"loss={initial_loss:.6e} "
            f"|grad|={np.linalg.norm(initial_gradient):.6e} "
            f"target_mean={initial_target_mean:.3f} Pa "
            f"background_ref={initial_background_mean:.3f} Pa "
            f"contrast={initial_target_mean / initial_background_mean:.4f}"
        )

        def objective(values):
            loss, gradient, _ = self.loss_and_gradient(values)
            return loss, gradient

        def callback(values):
            loss, gradient, field = self.loss_and_gradient(values)
            target_mean, background_mean = self.loss_model.stats(field)
            state = PhaseOptimizationState(
                iteration=len(history) + 1,
                loss=loss,
                gradient_norm=float(np.linalg.norm(gradient)),
                target_mean=target_mean,
                background_mean=background_mean,
            )
            history.append(state)
            print(
                "[PhaseOnly] "
                f"iter={state.iteration:04d} "
                f"loss={state.loss:.6e} "
                f"|grad|={state.gradient_norm:.6e} "
                f"target_mean={state.target_mean:.3f} Pa "
                f"background_ref={state.background_mean:.3f} Pa "
                f"contrast={state.target_mean / state.background_mean:.4f}"
            )

        result = minimize(
            fun=objective,
            x0=phases,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 2.0 * np.pi)] * phases.size,
            callback=callback,
            options={
                "maxiter": iterations,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "maxls": 40,
            },
        )
        phases = np.mod(result.x, 2.0 * np.pi)
        return phases, history

    def gradient_check(self, phases: np.ndarray, max_checks: int = 8) -> list[dict]:
        phases = np.asarray(phases, dtype=np.float64)
        loss, gradient, _ = self.loss_and_gradient(phases)
        step = float(self.cfg.training.gradient_check_step)
        count = min(int(max_checks), phases.size)
        rows = []
        for index in range(count):
            offset = np.zeros_like(phases)
            offset[index] = step
            loss_plus, _, _ = self.loss_and_gradient(phases + offset)
            loss_minus, _, _ = self.loss_and_gradient(phases - offset)
            finite_difference = (loss_plus - loss_minus) / (2.0 * step)
            absolute_error = abs(finite_difference - gradient[index])
            denominator = max(abs(finite_difference), abs(gradient[index]), 1.0)
            rows.append(
                {
                    "index": index,
                    "loss": loss,
                    "adjoint_gradient": float(gradient[index]),
                    "finite_difference": float(finite_difference),
                    "absolute_error": float(absolute_error),
                    "relative_error": float(absolute_error / denominator),
                }
            )
        return rows
