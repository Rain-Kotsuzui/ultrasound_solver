import time

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import scipy.sparse as sp
import warp as wp
from nvmath.sparse.advanced import DirectSolver
from pypardiso import PyPardisoSolver

from boundary_condensed_solver import BoundaryCondensedDirectSolver
from config import SimulationConfig
from iterative_solver import GpuIterativeSolver
from schur_direct_solver import RecursiveGridSchurDirectSolver
from transducer_array import TransducerArray
from warp_utils import WarpAssemblyEngine


class HelmholtzDirectSolver:
    """Assemble and solve the discrete Helmholtz system."""

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.transducers = TransducerArray(config)
        self.total_dofs = (
            self.cfg.domain.nx
            * self.cfg.domain.ny
            * self.cfg.domain.nz
        )
        self.warp_engine = WarpAssemblyEngine(
            nx=self.cfg.domain.nx,
            ny=self.cfg.domain.ny,
            nz=self.cfg.domain.nz,
            device="cuda",
        )

        self.solve_backend = "uninitialized"
        self.A_cached_csr_cpu = None
        self.A_cached_csr_gpu = None
        self.A_cached_H_csr_gpu = None
        self.rhs_cached = None
        self.condensed_solver = None
        self.schur_solver = None
        self.iterative_solver = GpuIterativeSolver(
            restart=int(self.cfg.solver.get("gmres_restart", 30)),
            maxiter=int(self.cfg.solver.get("gmres_maxiter", 3000)),
            rtol=float(self.cfg.solver.get("gmres_rtol", 1.0e-6)),
            backward_error_tol=float(
                self.cfg.solver.get(
                    "gmres_backward_error_tol",
                    1.0e-4,
                )
            ),
        )
        self.pardiso_solver = PyPardisoSolver(mtype=11)

    def assemble_matrix(self, force: bool = False) -> np.ndarray:
        if self.A_cached_csr_cpu is not None and not force:
            return self.rhs_cached
        if force:
            self.invalidate_factorization()

        obstacle = self.cfg.obstacles[0] if self.cfg.obstacles else {}
        matrix, _ = self.warp_engine.assemble_system_gpu(
            dx=self.cfg.domain.dx,
            omega=self.cfg.omega,
            k0_bg=self.cfg.k0,
            rho0=self.cfg.physics.medium_density,
            c0=self.cfg.physics.sound_speed,
            obs_cfg=obstacle,
            trans_phases_np=self.transducers.phases,
            trans_amps_np=self.transducers.amplitudes,
            array_n=self.cfg.specs.array_n,
            trans_radius=self.cfg.specs.diameter * 0.5,
            trans_pitch=self.cfg.specs.pitch,
            bcs_dict=self.cfg.boundary_conditions,
        )
        self.A_cached_csr_cpu = matrix.tocsr()
        return self.update_rhs()

    def update_rhs(self) -> np.ndarray:
        self.rhs_cached = self.warp_engine.assemble_rhs(
            trans_phases_np=self.transducers.phases,
            trans_amps_np=self.transducers.amplitudes,
        )
        return self.rhs_cached

    def set_phases(self, phases: np.ndarray):
        self.transducers.set_phases(phases)

    def invalidate_factorization(self):
        if self.condensed_solver is not None:
            self.condensed_solver.close()
        self.condensed_solver = None
        self.schur_solver = None
        self.A_cached_csr_cpu = None
        self.A_cached_csr_gpu = None
        self.A_cached_H_csr_gpu = None
        self.rhs_cached = None
        self.solve_backend = "uninitialized"

    def factorize(self):
        if self.A_cached_csr_cpu is None:
            self.assemble_matrix()
        backend = self._requested_backend()
        if backend == "cudss_hybrid_direct":
            self._ensure_condensed_solver()
        elif backend == "nested_schur_direct":
            self._ensure_schur_solver()

    def solve(self) -> np.ndarray:
        print("\n" + "=" * 65)
        print(
            "[Smart Solver] 启动智能混合线性求解 "
            f"(DOFs: {self.total_dofs:,})"
        )
        print("=" * 65)

        start = time.perf_counter()
        reuse_matrix = bool(self.cfg.solver.get("reuse_matrix", True))
        if self.A_cached_csr_cpu is not None and reuse_matrix:
            rhs = self.update_rhs()
            stage = "相位 RHS 更新"
        else:
            rhs = self.assemble_matrix(
                force=self.A_cached_csr_cpu is not None
            )
            stage = "Warp GPU 矩阵装配"
        assembly_time = time.perf_counter() - start
        print(
            f"[OK] 阶段 1/2: {stage}完成 | "
            f"NNZ: {self.A_cached_csr_cpu.nnz:,} | "
            f"耗时: {assembly_time * 1000.0:.2f} ms"
        )

        start = time.perf_counter()
        solution = self.solve_rhs(rhs)
        print(
            "[OK] 阶段 2/2: 线性系统求解完成! "
            f"耗时: {time.perf_counter() - start:.2f} 秒"
        )
        print("=" * 65)
        return self._reshape_solution(solution)

    def solve_rhs(self, rhs: np.ndarray) -> np.ndarray:
        if self.A_cached_csr_cpu is None:
            raise RuntimeError("矩阵尚未装配，请先调用 assemble_matrix()")

        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        backend = self._requested_backend()

        if backend == "cudss_hybrid_direct":
            print("[RUN] 阶段 2/2: 调用凝聚复对称 cuDSS 直接求解...")
            self._ensure_condensed_solver()
            solution = self.condensed_solver.solve(rhs_matrix)
            self._validate_direct_residual(
                self.condensed_solver.last_relative_residual
            )
            stats = self.condensed_solver.stats()
            print(
                "  --> cuDSS: "
                f"mode={stats['execution_mode']} | "
                f"plan={stats['plan_seconds']:.3f}s | "
                f"factor={stats['factor_seconds']:.3f}s | "
                f"residual={stats['last_relative_residual']:.3e}"
            )
            self.solve_backend = "cudss_hybrid_direct"

        elif backend == "nested_schur_direct":
            print("[RUN] 阶段 2/2: 调用 Nested Schur 直接求解...")
            reuse = bool(
                self.cfg.solver.get("reuse_factorization", True)
            )
            self._ensure_schur_solver(force=not reuse)
            solution = self.schur_solver.solve(rhs_matrix)
            self.solve_backend = "nested_schur_direct"

        elif backend == "gpu_direct":
            print("[RUN] 阶段 2/2: 调用 NVIDIA cuDSS GPU 直接求解...")
            self._ensure_gpu_matrix()
            solution = cp.asnumpy(
                self._solve_gpu_direct(
                    self.A_cached_csr_gpu,
                    cp.asfortranarray(cp.asarray(rhs_matrix)),
                )
            )
            self.solve_backend = "gpu_direct"

        elif backend == "cpu_direct":
            print("[RUN] 阶段 2/2: 调用 Pardiso CPU 直接求解...")
            solution = self._solve_cpu_direct(
                self.A_cached_csr_cpu,
                rhs_matrix,
            )
            self.solve_backend = "cpu_direct"

        else:
            solution = self._solve_iterative_with_fallback(rhs_matrix)

        solution = np.asarray(solution)
        return solution[:, 0] if squeeze else solution

    def solve_adjoint(self, adjoint_rhs: np.ndarray) -> np.ndarray:
        if self.A_cached_csr_cpu is None:
            self.assemble_matrix()
        rhs_matrix, squeeze = self._as_adjoint_rhs_matrix(adjoint_rhs)
        backend = self._requested_backend()
        if backend == "auto":
            backend = (
                "gpu_iterative"
                if self.solve_backend == "uninitialized"
                else self.solve_backend
            )

        if backend == "cudss_hybrid_direct":
            self._ensure_condensed_solver()
            solution = self.condensed_solver.solve_adjoint(rhs_matrix)
            self._validate_direct_residual(
                self.condensed_solver.last_adjoint_relative_residual
            )
        elif backend == "nested_schur_direct":
            self._ensure_schur_solver()
            solution = self.schur_solver.solve_adjoint(rhs_matrix)
        elif backend in {"gpu_iterative", "gpu_direct"}:
            solution = self._solve_gpu_adjoint(rhs_matrix, backend)
        else:
            matrix_h = (
                self.A_cached_csr_cpu
                .conjugate()
                .transpose()
                .tocsr()
            )
            solution = self._solve_cpu_direct(matrix_h, rhs_matrix)

        solution = np.asarray(solution)
        if squeeze:
            solution = solution[:, 0]
        return self._reshape_solution(solution)

    def _ensure_condensed_solver(self):
        if self.condensed_solver is not None:
            return
        fixed_dofs, _ = self.warp_engine.emitter_dof_map()
        self.condensed_solver = BoundaryCondensedDirectSolver(
            matrix=self.A_cached_csr_cpu,
            dims=(
                self.cfg.domain.nx,
                self.cfg.domain.ny,
                self.cfg.domain.nz,
            ),
            boundary_conditions=self.cfg.boundary_conditions,
            fixed_dofs=fixed_dofs,
            num_threads=int(
                self.cfg.solver.get("hybrid_num_threads", 1)
            ),
            multithreading_lib=self.cfg.solver.get(
                "hybrid_multithreading_lib"
            ),
            execution_mode=str(
                self.cfg.solver.get("hybrid_execution_mode", "auto")
            ),
            device_memory_limit_gb=float(
                self.cfg.solver.get(
                    "hybrid_device_memory_limit_gb",
                    5.6,
                )
            ),
            device_memory_reserve_gb=float(
                self.cfg.solver.get(
                    "hybrid_device_memory_reserve_gb",
                    0.75,
                )
            ),
            register_cuda_memory=bool(
                self.cfg.solver.get(
                    "hybrid_register_cuda_memory",
                    False,
                )
            ),
            symmetry_tolerance=float(
                self.cfg.solver.get(
                    "condensed_symmetry_tolerance",
                    1.0e-12,
                )
            ),
        )

    def _ensure_schur_solver(self, force: bool = False):
        if self.schur_solver is not None and not force:
            return
        self.schur_solver = RecursiveGridSchurDirectSolver.from_grid(
            matrix=self.A_cached_csr_cpu,
            nx=self.cfg.domain.nx,
            ny=self.cfg.domain.ny,
            nz=self.cfg.domain.nz,
            leaf_dofs=int(self.cfg.solver.get("leaf_dofs", 4096)),
            min_axis_size=int(
                self.cfg.solver.get("min_axis_size", 3)
            ),
            schur_block_size=int(
                self.cfg.solver.get("schur_block_size", 256)
            ),
            dense_backend=str(
                self.cfg.solver.get("dense_backend", "gpu")
            ),
            gpu_min_separator_dofs=int(
                self.cfg.solver.get(
                    "gpu_min_separator_dofs",
                    512,
                )
            ),
            parallel_depth=int(
                self.cfg.solver.get("parallel_depth", 1)
            ),
            profile_print_depth=int(
                self.cfg.solver.get("profile_print_depth", 0)
            ),
        )

    def _solve_iterative_with_fallback(
        self,
        rhs_matrix: np.ndarray,
    ) -> np.ndarray:
        if rhs_matrix.shape[1] != 1:
            raise ValueError("gpu_iterative supports one RHS at a time")
        self._ensure_gpu_matrix()
        rhs_gpu = cp.asarray(rhs_matrix[:, 0])
        print("[RUN] 阶段 2/2: 调用 GPU Jacobi-GMRES...")
        try:
            solution = self.iterative_solver.solve(
                self.A_cached_csr_gpu,
                rhs_gpu,
            )
            self.solve_backend = "gpu_iterative"
            return cp.asnumpy(solution).reshape((-1, 1))
        except RuntimeError as error:
            print(f"  [WARN] GPU 迭代法未通过校验: {error}")
            if self._estimate_vram_and_decide_backend():
                solution = self._solve_gpu_direct(
                    self.A_cached_csr_gpu,
                    rhs_gpu,
                )
                self.solve_backend = "gpu_direct"
                return cp.asnumpy(solution).reshape((-1, 1))
            self.solve_backend = "cpu_direct"
            return self._solve_cpu_direct(
                self.A_cached_csr_cpu,
                rhs_matrix,
            )

    def _solve_gpu_adjoint(
        self,
        rhs_matrix: np.ndarray,
        backend: str,
    ) -> np.ndarray:
        if rhs_matrix.shape[1] != 1:
            raise ValueError(f"{backend} supports one adjoint RHS")
        if self.A_cached_H_csr_gpu is None:
            matrix_h = (
                self.A_cached_csr_cpu
                .conjugate()
                .transpose()
                .tocsr()
            )
            self.A_cached_H_csr_gpu = csp.csr_matrix(matrix_h)
        rhs_gpu = cp.asarray(rhs_matrix[:, 0])
        if backend == "gpu_direct":
            result = self._solve_gpu_direct(
                self.A_cached_H_csr_gpu,
                rhs_gpu,
            )
        else:
            result = self.iterative_solver.solve(
                self.A_cached_H_csr_gpu,
                rhs_gpu,
            )
        return cp.asnumpy(result).reshape((-1, 1))

    def _requested_backend(self) -> str:
        backend = str(
            self.cfg.solver.get("backend", "gpu_iterative")
        ).lower()
        valid = {
            "auto",
            "gpu_iterative",
            "cudss_hybrid_direct",
            "nested_schur_direct",
            "gpu_direct",
            "cpu_direct",
        }
        if backend not in valid:
            raise ValueError(
                f"未知 solver.backend={backend!r}，可选值为 "
                f"{sorted(valid)}"
            )
        return backend

    def _ensure_gpu_matrix(self):
        if self.A_cached_csr_gpu is None:
            self.A_cached_csr_gpu = csp.csr_matrix(
                self.A_cached_csr_cpu
            )

    def _validate_direct_residual(self, residual: float):
        tolerance = float(
            self.cfg.solver.get("direct_residual_tol", 1.0e-4)
        )
        if not np.isfinite(residual) or residual > tolerance:
            raise RuntimeError(
                "Direct solve failed validation: "
                f"relative residual={residual:.3e}, "
                f"tolerance={tolerance:.3e}"
            )

    def _as_rhs_matrix(
        self,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        array = np.asarray(rhs, dtype=np.complex128)
        if array.ndim == 1 and array.size == self.total_dofs:
            return np.asfortranarray(array.reshape((-1, 1))), True
        if array.ndim == 2 and array.shape[0] == self.total_dofs:
            return np.asfortranarray(array), False
        raise ValueError(
            f"rhs must have shape ({self.total_dofs},) or "
            f"({self.total_dofs}, nrhs), got {array.shape}"
        )

    def _as_adjoint_rhs_matrix(
        self,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        array = np.asarray(rhs, dtype=np.complex128)
        spatial_shape = (
            self.cfg.domain.nx,
            self.cfg.domain.ny,
            self.cfg.domain.nz,
        )
        if array.shape == spatial_shape:
            return np.asfortranarray(
                array.reshape((-1, 1), order="F")
            ), True
        if array.ndim == 4 and array.shape[:3] == spatial_shape:
            return np.asfortranarray(
                array.reshape((self.total_dofs, -1), order="F")
            ), False
        return self._as_rhs_matrix(array)

    def _reshape_solution(self, solution: np.ndarray) -> np.ndarray:
        array = np.asarray(solution)
        spatial_shape = (
            self.cfg.domain.nx,
            self.cfg.domain.ny,
            self.cfg.domain.nz,
        )
        if array.ndim == 1:
            return array.reshape(spatial_shape, order="F")
        return array.reshape(
            spatial_shape + (array.shape[1],),
            order="F",
        )

    def _solve_cpu_direct(
        self,
        matrix: sp.spmatrix,
        rhs: np.ndarray,
    ) -> np.ndarray:
        real_matrix = sp.bmat(
            [
                [matrix.real, -matrix.imag],
                [matrix.imag, matrix.real],
            ],
            format="csr",
        )
        result = np.empty(rhs.shape, dtype=np.complex128, order="F")
        for column in range(rhs.shape[1]):
            real_rhs = np.concatenate([
                rhs[:, column].real,
                rhs[:, column].imag,
            ])
            real_solution = self.pardiso_solver.solve(
                real_matrix,
                real_rhs,
            )
            result[:, column] = (
                real_solution[: self.total_dofs]
                + 1j * real_solution[self.total_dofs :]
            )
        return result

    @staticmethod
    def _solve_gpu_direct(matrix, rhs):
        with DirectSolver(matrix, rhs) as solver:
            solver.plan()
            solver.factorize()
            solution = solver.solve()
        residual = cp.linalg.norm(
            matrix @ solution - rhs
        ) / cp.linalg.norm(rhs)
        residual = float(residual.get())
        if (
            not bool(cp.isfinite(solution).all().get())
            or residual > 1.0e-4
        ):
            raise RuntimeError(
                "GPU direct solve failed validation: "
                f"relative residual={residual:.3e}"
            )
        return solution

    def _estimate_vram_and_decide_backend(self) -> bool:
        nnz_lu = 1.2 * (self.total_dofs ** (4.0 / 3.0))
        lu_gib = nnz_lu * 20.0 / (1024 ** 3)
        required_gib = lu_gib * 8.5 + 0.5
        free_bytes, _ = cp.cuda.Device(0).mem_info
        return required_gib + 1.0 < free_bytes / (1024 ** 3)

    def to_warp_array(self, np_field: np.ndarray) -> wp.array:
        return wp.from_numpy(
            np_field,
            dtype=wp.complex64,
            device=self.warp_engine.device,
        )
