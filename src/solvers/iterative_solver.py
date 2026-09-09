import cupy as cp
import cupyx.scipy.sparse.linalg as csplinalg


class GpuIterativeSolver:
    """GPU row-scaled Jacobi-GMRES solver for sparse complex systems."""

    def __init__(
        self,
        restart: int = 30,
        maxiter: int = 3000,
        rtol: float = 1.0e-6,
        backward_error_tol: float = 1.0e-4,
    ):
        self.restart = int(restart)
        self.maxiter = int(maxiter)
        self.rtol = float(rtol)
        self.backward_error_tol = float(backward_error_tol)
        self.last_iterations = 0
        self.last_scaled_residual = None
        self.last_backward_error = None

    def solve(self, A_gpu, rhs_gpu):
        if float(cp.linalg.norm(rhs_gpu).get()) == 0.0:
            self.last_iterations = 0
            self.last_scaled_residual = 0.0
            self.last_backward_error = 0.0
            return cp.zeros_like(rhs_gpu)

        row_max = cp.asarray(abs(A_gpu).max(axis=1).toarray()).ravel()
        row_scale = cp.where(row_max > 0.0, 1.0 / row_max, 1.0)
        scaled_rhs = row_scale * rhs_gpu
        scaled_operator = csplinalg.LinearOperator(
            A_gpu.shape,
            matvec=lambda vector: row_scale * (A_gpu @ vector),
            dtype=A_gpu.dtype,
        )

        scaled_diagonal = row_scale * A_gpu.diagonal()
        inverse_diagonal = cp.where(
            cp.abs(scaled_diagonal) > 0.0,
            1.0 / scaled_diagonal,
            1.0,
        )
        preconditioner = csplinalg.LinearOperator(
            A_gpu.shape,
            matvec=lambda vector: inverse_diagonal * vector,
            dtype=A_gpu.dtype,
        )

        residuals = []
        solution, info = csplinalg.gmres(
            scaled_operator,
            scaled_rhs,
            M=preconditioner,
            restart=self.restart,
            maxiter=self.maxiter,
            rtol=self.rtol,
            atol=0.0,
            callback=residuals.append,
            callback_type="pr_norm",
        )
        scaled_residual = cp.linalg.norm(
            scaled_operator @ solution - scaled_rhs
        ) / cp.linalg.norm(scaled_rhs)
        scaled_residual = float(scaled_residual.get())
        residual = A_gpu @ solution - rhs_gpu
        denominator = abs(A_gpu) @ cp.abs(solution) + cp.abs(rhs_gpu)
        backward_error = cp.max(
            cp.abs(residual) / cp.maximum(denominator, 1.0e-30)
        )
        backward_error = float(backward_error.get())
        is_finite = bool(cp.isfinite(solution).all().get())
        iterations = len(residuals) * self.restart

        self.last_iterations = iterations
        self.last_scaled_residual = scaled_residual
        self.last_backward_error = backward_error

        if (
            info != 0
            or not is_finite
            or scaled_residual > self.rtol
            or backward_error > self.backward_error_tol
        ):
            raise RuntimeError(
                f"info={info}, iterations={iterations}, "
                f"scaled residual={scaled_residual:.3e}, "
                f"backward error={backward_error:.3e}"
            )

        print(
            f"  --> GMRES 迭代次数: {iterations} | "
            f"缩放相对残差: {scaled_residual:.3e} | "
            f"后向误差: {backward_error:.3e}"
        )
        return solution
