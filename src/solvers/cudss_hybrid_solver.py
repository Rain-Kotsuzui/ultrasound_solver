import logging
import time

import cupy as cp
import numpy as np
import scipy.sparse as sp
from nvmath.sparse.advanced import DirectSolver
from nvmath.sparse.advanced import DirectSolverMatrixType
from nvmath.sparse.advanced import DirectSolverMatrixViewType
from nvmath.sparse.advanced import DirectSolverReorderingAlg


class CuDSSHybridSymmetricSolver:
    """Stateful exact cuDSS solver for a lower complex-symmetric matrix."""

    _MEMORY_FIELDS = (
        "permanent_device_memory",
        "peak_device_memory",
        "permanent_host_memory",
        "peak_host_memory",
        "hybrid_min_device_memory",
        "hybrid_max_device_memory",
    )

    def __init__(
        self,
        matrix: sp.spmatrix,
        num_threads: int = 1,
        multithreading_lib: str = None,
        execution_mode: str = "auto",
        device_memory_limit_gb: float = 5.6,
        device_memory_reserve_gb: float = 0.75,
        register_cuda_memory: bool = False,
    ):
        self.matrix = matrix.tocsr()
        self.n = int(self.matrix.shape[0])
        self.num_threads = int(num_threads)
        self.multithreading_lib = multithreading_lib
        self.requested_execution_mode = str(execution_mode).lower()
        self.device_memory_limit_gb = float(device_memory_limit_gb)
        self.device_memory_reserve_gb = float(device_memory_reserve_gb)
        self.register_cuda_memory = bool(register_cuda_memory)
        self.rhs_buffer = np.zeros(self.n, dtype=self.matrix.dtype)
        self.solver = None
        self.execution_mode = None
        self.plan_time = 0.0
        self.factor_time = 0.0
        self.solve_time = 0.0
        self.solve_count = 0
        self.factor_nnz = None
        self.memory_estimates = None
        self.available_device_memory = 0
        self.device_memory_budget = 0
        self.last_relative_residual = None
        self._validate_options()
        self._factorize()

    def _validate_options(self):
        valid_modes = {"auto", "hybrid_execute", "hybrid_memory"}
        if self.requested_execution_mode not in valid_modes:
            raise ValueError(
                f"execution_mode must be one of {sorted(valid_modes)}, "
                f"got {self.requested_execution_mode!r}"
            )
        if self.num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        if self.device_memory_limit_gb <= 0.0:
            raise ValueError("device_memory_limit_gb must be positive")
        if self.device_memory_reserve_gb < 0.0:
            raise ValueError("device_memory_reserve_gb cannot be negative")

    def _factorize(self):
        free_memory, _ = cp.cuda.Device(0).mem_info
        configured_limit = self._gib_to_bytes(self.device_memory_limit_gb)
        reserve = self._gib_to_bytes(self.device_memory_reserve_gb)
        self.available_device_memory = int(free_memory)
        self.device_memory_budget = min(
            configured_limit,
            max(0, self.available_device_memory - reserve),
        )

        initial_mode = self.requested_execution_mode
        if initial_mode == "auto":
            initial_mode = "hybrid_execute"
        self._plan(initial_mode)

        if initial_mode == "hybrid_execute":
            required = self.memory_estimates["peak_device_memory"]
            if required > self.device_memory_budget:
                if self.requested_execution_mode == "hybrid_execute":
                    self._raise_memory_error(required, "hybrid execute")
                self._plan("hybrid_memory")

        if self.execution_mode == "hybrid_memory":
            hybrid_minimum = self.memory_estimates[
                "hybrid_min_device_memory"
            ]
            if hybrid_minimum > self.device_memory_budget:
                self._raise_memory_error(
                    hybrid_minimum,
                    "hybrid memory minimum",
                )

        start = time.perf_counter()
        factor_info = self.solver.factorize()
        self.factor_time = time.perf_counter() - start
        self.factor_nnz = int(factor_info.lu_nnz)

    def _plan(self, execution_mode: str):
        if self.solver is not None:
            self.solver.free()
            self.solver = None

        logger = logging.getLogger("cudss.hybrid")
        logger.setLevel(logging.ERROR)
        options = {
            "blocking": True,
            "logger": logger,
            "sparse_system_type": DirectSolverMatrixType.SYMMETRIC,
            "sparse_system_view": DirectSolverMatrixViewType.LOWER,
        }
        if self.multithreading_lib is not None:
            options["multithreading_lib"] = self.multithreading_lib

        if execution_mode == "hybrid_execute":
            execution = {
                "name": "hybrid",
                "device_id": 0,
                "num_threads": self.num_threads,
            }
        else:
            execution = {
                "name": "cuda",
                "device_id": 0,
                "hybrid_memory_mode_options": {
                    "hybrid_memory_mode": True,
                    "hybrid_device_memory_limit": self.device_memory_budget,
                    "register_cuda_memory": self.register_cuda_memory,
                },
            }

        self.solver = DirectSolver(
            self.matrix,
            self.rhs_buffer,
            options=options,
            execution=execution,
        )
        self.solver.plan_config.reordering_algorithm = (
            DirectSolverReorderingAlg.NESTED_DISSECTION
        )

        start = time.perf_counter()
        plan_info = self.solver.plan()
        self.plan_time += time.perf_counter() - start
        self.execution_mode = execution_mode
        self.memory_estimates = self._memory_estimates_dict(
            plan_info.memory_estimates
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        result = np.empty(rhs_matrix.shape, dtype=self.matrix.dtype, order="F")
        start = time.perf_counter()
        for column in range(rhs_matrix.shape[1]):
            self.rhs_buffer[:] = rhs_matrix[:, column]
            self.solver.reset_operands(b=self.rhs_buffer)
            result[:, column] = self.solver.solve()
        self.solve_time += time.perf_counter() - start
        self.solve_count += rhs_matrix.shape[1]
        self.last_relative_residual = self.relative_residual(
            result,
            rhs_matrix,
        )
        return result[:, 0] if squeeze else result

    def relative_residual(
        self,
        solution: np.ndarray,
        rhs: np.ndarray,
    ) -> float:
        residual_norm = np.linalg.norm(
            self._matrix_product(solution) - rhs
        )
        rhs_norm = np.linalg.norm(rhs)
        return float(residual_norm / rhs_norm) if rhs_norm > 0.0 else 0.0

    def _matrix_product(self, value: np.ndarray) -> np.ndarray:
        product = (
            self.matrix @ value
            + self.matrix.transpose() @ value
        )
        diagonal = self.matrix.diagonal()
        if value.ndim == 2:
            diagonal = diagonal[:, None]
        return product - diagonal * value

    def stats(self) -> dict:
        return {
            "dofs": self.n,
            "nnz": int(self.matrix.nnz),
            "plan_seconds": self.plan_time,
            "factor_seconds": self.factor_time,
            "solve_seconds": self.solve_time,
            "solve_count": self.solve_count,
            "factor_nnz": self.factor_nnz,
            "last_relative_residual": self.last_relative_residual,
            "requested_execution_mode": self.requested_execution_mode,
            "execution_mode": self.execution_mode,
            "num_threads": self.num_threads,
            "available_device_memory": self.available_device_memory,
            "device_memory_budget": self.device_memory_budget,
            "memory_estimates": self.memory_estimates,
        }

    def close(self):
        if self.solver is not None:
            self.solver.free()
            self.solver = None

    def _raise_memory_error(self, required: int, requirement_name: str):
        host_detail = ""
        if self.execution_mode == "hybrid_memory":
            host_memory = self.memory_estimates["peak_host_memory"]
            host_detail = (
                "The symbolic estimate also requires up to "
                f"{self._format_gib(host_memory)} host memory for factors. "
            )
        self.close()
        raise MemoryError(
            "cuDSS exact factorization exceeds the configured GPU budget: "
            f"{requirement_name}={self._format_gib(required)}, "
            f"budget={self._format_gib(self.device_memory_budget)}, "
            f"currently free={self._format_gib(self.available_device_memory)}. "
            f"{host_detail}"
            "Reduce the grid, increase GPU memory, or select another exact "
            "domain-decomposition backend."
        )

    @classmethod
    def _memory_estimates_dict(cls, estimates) -> dict:
        return {
            field: int(estimates[field])
            for field in cls._MEMORY_FIELDS
        }

    @staticmethod
    def _gib_to_bytes(value: float) -> int:
        return int(value * (1024 ** 3))

    @staticmethod
    def _format_gib(value: int) -> str:
        return f"{value / (1024 ** 3):.2f} GiB"

    def _as_rhs_matrix(
        self,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        rhs_array = np.asarray(rhs, dtype=self.matrix.dtype)
        if rhs_array.ndim == 1:
            if rhs_array.size != self.n:
                raise ValueError(
                    f"rhs has {rhs_array.size} entries, expected {self.n}"
                )
            return rhs_array.reshape((-1, 1)), True
        if rhs_array.ndim == 2 and rhs_array.shape[0] == self.n:
            return np.asfortranarray(rhs_array), False
        raise ValueError(
            f"rhs must have shape ({self.n},) or ({self.n}, nrhs), "
            f"got {rhs_array.shape}"
        )
