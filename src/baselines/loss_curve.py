"""Non-blocking live loss curve shared by all phase optimization algorithms."""

import warnings

import numpy as np


class LiveLossCurve:
    def __init__(self, enabled: bool, update_interval: int, pause_seconds: float):
        self.enabled = bool(enabled)
        self.update_interval = int(update_interval)
        self.pause_seconds = float(pause_seconds)
        self.evaluations = []
        self.losses = []
        self.best_losses = []
        self._plt = None
        self._figure = None
        self._axis = None
        self._loss_line = None
        self._best_line = None

        if not self.enabled:
            return
        if self.update_interval < 1:
            raise ValueError("loss_curve_update_interval must be at least 1")
        if not np.isfinite(self.pause_seconds) or self.pause_seconds < 0:
            raise ValueError("loss_curve_pause_seconds must be finite and nonnegative")
        try:
            import matplotlib.pyplot as plt
            from matplotlib import rcsetup

            backend = plt.get_backend().lower()
            non_interactive = {
                name.lower() for name in rcsetup.non_interactive_bk
            }
            if backend in non_interactive:
                self.enabled = False
                warnings.warn(
                    f"Live loss curve disabled: {backend} is not an interactive backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return

            plt.ion()
            self._plt = plt
            self._figure, self._axis = plt.subplots(
                figsize=(8.5, 5.0), num="Ultrasound Phase Optimization Loss"
            )
            self._figure.canvas.manager.set_window_title(
                "Ultrasound Phase Optimization Loss"
            )
            self._loss_line, = self._axis.plot(
                [], [], color="#2563eb", linewidth=1.6, alpha=0.72,
                label="Current loss",
            )
            self._best_line, = self._axis.plot(
                [], [], color="#dc2626", linewidth=2.1, label="Best loss",
            )
            self._axis.set_title("Live Phase Optimization Loss")
            self._axis.set_xlabel("Field evaluations")
            self._axis.set_ylabel("Loss")
            self._axis.grid(True, alpha=0.28)
            self._axis.legend(loc="best")
            self._figure.tight_layout()
            self._figure.show()
            self._draw()
        except Exception as exc:
            self.enabled = False
            warnings.warn(
                f"Live loss curve disabled: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def update(self, row: dict):
        if not self.enabled:
            return
        self.evaluations.append(int(row["evaluation"]))
        self.losses.append(float(row["loss"]))
        self.best_losses.append(float(row["best_loss"]))
        if len(self.evaluations) % self.update_interval == 0:
            self._draw()

    def finalize(self, termination_reason: str):
        if not self.enabled:
            return
        self._draw()
        self._axis.set_title(
            f"Phase Optimization Loss ({termination_reason})"
        )
        self._figure.canvas.draw_idle()
        self._flush()

    def _draw(self):
        if not self.evaluations:
            return
        self._loss_line.set_data(self.evaluations, self.losses)
        self._best_line.set_data(self.evaluations, self.best_losses)
        self._axis.relim()
        self._axis.autoscale_view()
        self._figure.canvas.draw_idle()
        self._flush()

    def _flush(self):
        try:
            self._figure.canvas.flush_events()
            self._plt.pause(self.pause_seconds)
        except Exception:
            # GUI backends can be closed by the user while the optimization continues.
            self.enabled = False
