"""Live loss views with either one standalone chart or one comparison dashboard."""

import math
import warnings

import numpy as np


def save_loss_history(history, output_path, title):
    """Write a durable, headless-safe PNG from recorded evaluation history."""
    if not history:
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = str(output_path)
    evaluations = [row["evaluation"] for row in history]
    losses = [row["loss"] for row in history]
    best_losses = [row["best_loss"] for row in history]
    figure = Figure(figsize=(8.5, 5.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    marker = "o" if len(evaluations) == 1 else None
    axis.plot(evaluations, losses, color="#2563eb", linewidth=1.3,
              alpha=0.72, marker=marker, label="Current loss")
    axis.plot(evaluations, best_losses, color="#dc2626", linewidth=2.0,
              marker=marker, label="Best loss")
    axis.set_title(title)
    axis.set_xlabel("Field evaluations")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.28)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=180)
    return output_path


def save_loss_dashboard(histories, output_path):
    """Write all completed algorithms into one static loss-dashboard PNG."""
    if not histories:
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    names = list(histories)
    columns = math.ceil(math.sqrt(len(names)))
    rows = math.ceil(len(names) / columns)
    figure = Figure(
        figsize=(5.2 * columns, 3.5 * rows),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    for index, name in enumerate(names):
        axis = figure.add_subplot(rows, columns, index + 1)
        history = histories[name]
        evaluations = [row["evaluation"] for row in history]
        marker = "o" if len(evaluations) == 1 else None
        axis.plot(
            evaluations, [row["loss"] for row in history],
            color="#2563eb", linewidth=1.1, alpha=0.72, marker=marker,
            label="Current",
        )
        axis.plot(
            evaluations, [row["best_loss"] for row in history],
            color="#dc2626", linewidth=1.7, marker=marker, label="Best",
        )
        axis.set_title(name)
        axis.set_xlabel("Field evaluations")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.28)
        axis.legend(loc="best", fontsize=8)
    figure.savefig(str(output_path), dpi=180)
    return str(output_path)


def _interactive_pyplot():
    import matplotlib.pyplot as plt

    backend = plt.get_backend().lower()
    non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    try:
        from matplotlib.backends.registry import BackendFilter, backend_registry
        non_interactive.update(
            backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
        )
    except ImportError:
        pass
    if backend in non_interactive:
        raise RuntimeError(f"{backend} is not an interactive backend")
    return plt


def _keep_window_passive(figure):
    """Do not request foreground focus when a GUI window is mapped."""
    manager = figure.canvas.manager
    window = getattr(manager, "window", None)
    if window is not None:
        try:
            window.attributes("-topmost", False)
        except Exception:
            pass


class ComparisonLossDashboard:
    """One fixed grid of loss plots for a multi-algorithm comparison."""

    def __init__(
        self,
        algorithms,
        enabled=True,
        pause_seconds=0.01,
        tensorboard_log_dir=None,
    ):
        self.enabled = bool(enabled)
        self.pause_seconds = float(pause_seconds)
        self._plt = None
        self._figure = None
        self._axes = {}
        self._curves = {}
        self._algorithms = list(algorithms)
        self._writer = None
        if tensorboard_log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(tensorboard_log_dir))
            except Exception as exc:
                warnings.warn(
                    f"TensorBoard loss logging disabled: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not self.enabled:
            return
        if not np.isfinite(self.pause_seconds) or self.pause_seconds < 0:
            raise ValueError("loss_curve_pause_seconds must be finite and nonnegative")
        try:
            self._plt = _interactive_pyplot()
            count = max(1, len(self._algorithms))
            columns = math.ceil(math.sqrt(count))
            rows = math.ceil(count / columns)
            self._figure, axes = self._plt.subplots(
                rows, columns, figsize=(5.2 * columns, 3.4 * rows),
                num="Ultrasound Algorithm Comparison Loss", clear=True,
                squeeze=False,
            )
            self._figure.canvas.manager.set_window_title(
                "Ultrasound Algorithm Comparison Loss"
            )
            flat_axes = list(np.asarray(axes, dtype=object).reshape(-1))
            for axis, algorithm in zip(flat_axes, self._algorithms):
                self._axes[algorithm] = axis
                axis.set_title(algorithm)
                axis.set_xlabel("Field evaluations")
                axis.set_ylabel("Loss")
                axis.grid(True, alpha=0.28)
            for axis in flat_axes[len(self._algorithms):]:
                axis.set_visible(False)
            self._figure.tight_layout()
            _keep_window_passive(self._figure)
            # Map once; subsequent loss updates only redraw this existing canvas.
            self._figure.show()
        except Exception as exc:
            self.enabled = False
            warnings.warn(
                f"Comparison loss dashboard disabled: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def curve(self, algorithm, update_interval):
        if algorithm not in self._axes:
            return LiveLossCurve(False, update_interval, self.pause_seconds, algorithm)
        curve = DashboardLossCurve(
            self, self._axes[algorithm], algorithm, update_interval
        )
        self._curves[algorithm] = curve
        return curve

    def flush(self):
        if not self.enabled or self._figure is None:
            return
        try:
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
            _keep_window_passive(self._figure)
        except Exception:
            self.enabled = False

    def log_tensorboard(self, algorithm, row):
        if self._writer is None:
            return
        step = int(row["evaluation"])
        self._writer.add_scalar(f"{algorithm}/loss_current", row["loss"], step)
        self._writer.add_scalar(f"{algorithm}/loss_best", row["best_loss"], step)
        self._writer.add_scalar(
            f"{algorithm}/target_mean", row["target_mean"], step
        )
        self._writer.add_scalar(
            f"{algorithm}/background_max", row["background_max"], step
        )

    def close(self):
        if self._writer is not None:
            self._writer.close()


class DashboardLossCurve:
    def __init__(self, dashboard, axis, title, update_interval):
        self.dashboard = dashboard
        self.axis = axis
        self.title = title
        self.update_interval = int(update_interval)
        self.enabled = dashboard.enabled and self.update_interval >= 1
        self.evaluations = []
        self.losses = []
        self.best_losses = []
        if self.enabled:
            self.loss_line, = axis.plot(
                [], [], color="#2563eb", linewidth=1.4, alpha=0.72,
                label="Current",
            )
            self.best_line, = axis.plot(
                [], [], color="#dc2626", linewidth=1.9, label="Best",
            )
            axis.legend(loc="best", fontsize=8)

    def update(self, row):
        if not self.enabled:
            return
        self.evaluations.append(int(row["evaluation"]))
        self.losses.append(float(row["loss"]))
        self.best_losses.append(float(row["best_loss"]))
        self.dashboard.log_tensorboard(self.title, row)
        if len(self.evaluations) % self.update_interval == 0:
            self._draw()

    def finalize(self, termination_reason):
        if not self.enabled:
            return
        self._draw()
        self.axis.set_title(f"{self.title} ({termination_reason})")
        self.dashboard.flush()

    def _draw(self):
        self.loss_line.set_data(self.evaluations, self.losses)
        self.best_line.set_data(self.evaluations, self.best_losses)
        self.axis.relim()
        self.axis.autoscale_view()
        self.dashboard.flush()


class LiveLossCurve:
    """Standalone loss curve for a normal single-algorithm run."""

    def __init__(self, enabled, update_interval, pause_seconds, title="phase_optimization"):
        self.enabled = bool(enabled)
        self.update_interval = int(update_interval)
        self.pause_seconds = float(pause_seconds)
        self.title = title
        self.evaluations = []
        self.losses = []
        self.best_losses = []
        self._dashboard = None
        self._curve = None

    def _ensure(self):
        if not self.enabled or self._curve is not None:
            return
        self._dashboard = ComparisonLossDashboard(
            [self.title], enabled=True, pause_seconds=self.pause_seconds
        )
        self._curve = self._dashboard.curve(self.title, self.update_interval)
        self.enabled = self._curve.enabled

    def update(self, row):
        self._ensure()
        if self.enabled:
            self._curve.update(row)

    def finalize(self, termination_reason):
        self._ensure()
        if self.enabled:
            self._curve.finalize(termination_reason)
