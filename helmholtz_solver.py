import time

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import warp as wp
from nvmath.sparse.advanced import DirectSolver

from boundary_condensed_solver import BoundaryCondensedDirectSolver
from config import SimulationConfig
from iterative_solver import GpuIterativeSolver
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

        self.A_cached_csr_cpu = None
        self.A_cached_csr_gpu = None
        self.A_cached_H_csr_gpu = None
        self.rhs_cached = None
        self._obs_sdf = None
        self.condensed_solver = None
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

    @property
    def obstacle_sdf(self) -> np.ndarray | None:
        """(nx, ny, nz) SDF of the active mesh obstacle, or None."""
        return self._obs_sdf

    def assemble_matrix(self, force: bool = False) -> np.ndarray:
        if self.A_cached_csr_cpu is not None and not force:
            return self.rhs_cached
        if force:
            self.invalidate_factorization()
            self._obs_sdf = None

        if len(self.cfg.obstacles) > 1:
            raise NotImplementedError(
                "The solver currently supports one obstacle at a time"
            )
        obstacle = self.cfg.obstacles[0] if self.cfg.obstacles else {}
        obs_sdf = None
        if obstacle.get("type", "").lower() == "mesh":
            if self._obs_sdf is None:
                from mesh_to_sdf import mesh_to_sdf

                self._obs_sdf = mesh_to_sdf(
                    mesh_path=obstacle["file"],
                    grid_size=(
                        self.cfg.domain.nx,
                        self.cfg.domain.ny,
                        self.cfg.domain.nz,
                    ),
                    box_size=(
                        self.cfg.domain.lx,
                        self.cfg.domain.ly,
                        self.cfg.domain.lz,
                    ),
                    center_m=obstacle["center_m"],
                    rotation_deg=obstacle.get(
                        "rotation_deg", (0.0, 0.0, 0.0)
                    ),
                    origin=(0.0, 0.0, 0.0),
                    scale=float(obstacle.get("scale", 1.0)),
                )
            obs_sdf = self._obs_sdf

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
            obs_sdf=obs_sdf,
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
        self.A_cached_csr_cpu = None
        self.A_cached_csr_gpu = None
        self.A_cached_H_csr_gpu = None
        self.rhs_cached = None

    def factorize(self):
        if self.A_cached_csr_cpu is None:
            self.assemble_matrix()
        if self._requested_backend() == "cudss_hybrid_direct":
            self._ensure_condensed_solver()

    def build_phase_response_basis(self):
        from phase_response_basis import PhaseResponseBasis

        training = self.cfg.training
        return PhaseResponseBasis(
            solver=self,
            storage_path=training.phase_basis_file,
            rhs_batch_size=training.phase_basis_batch_size,
            voxel_chunk_size=training.voxel_chunk_size,
            release_factor_after_build=(
                training.release_factor_after_basis
            ),
        ).build()

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

    def solve_rhs(
        self,
        rhs: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        if self.A_cached_csr_cpu is None:
            raise RuntimeError("矩阵尚未装配，请先调用 assemble_matrix()")

        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        backend = self._requested_backend()

        if backend == "cudss_hybrid_direct":
            if verbose:
                print(
                    "[RUN] 阶段 2/2: "
                    "调用凝聚复对称 cuDSS 直接求解..."
                )
            self._ensure_condensed_solver()
            solution = self.condensed_solver.solve(rhs_matrix)
            self._validate_direct_residual(
                self.condensed_solver.last_relative_residual
            )
            if verbose:
                stats = self.condensed_solver.stats()
                print(
                    "  --> cuDSS: "
                    f"mode={stats['execution_mode']} | "
                    f"plan={stats['plan_seconds']:.3f}s | "
                    f"factor={stats['factor_seconds']:.3f}s | "
                    f"residual={stats['last_relative_residual']:.3e}"
                )

        elif backend == "gpu_direct":
            if verbose:
                print(
                    "[RUN] 阶段 2/2: "
                    "调用 NVIDIA cuDSS GPU 直接求解..."
                )
            self._ensure_gpu_matrix()
            gpu_rhs = (
                cp.asarray(rhs_matrix[:, 0])
                if squeeze
                else cp.asfortranarray(cp.asarray(rhs_matrix))
            )
            solution = cp.asnumpy(
                self._solve_gpu_direct(
                    self.A_cached_csr_gpu,
                    gpu_rhs,
                )
            )

        else:
            solution = self._solve_iterative(
                rhs_matrix,
                verbose=verbose,
            )

        solution = np.asarray(solution)
        if squeeze and solution.ndim == 2:
            return solution[:, 0]
        return solution

    def solve_adjoint(self, adjoint_rhs: np.ndarray) -> np.ndarray:
        if self.A_cached_csr_cpu is None:
            self.assemble_matrix()
        rhs_matrix, squeeze = self._as_adjoint_rhs_matrix(adjoint_rhs)
        backend = self._requested_backend()

        if backend == "cudss_hybrid_direct":
            self._ensure_condensed_solver()
            solution = self.condensed_solver.solve_adjoint(rhs_matrix)
            self._validate_direct_residual(
                self.condensed_solver.last_adjoint_relative_residual
            )
        else:
            solution = self._solve_gpu_adjoint(rhs_matrix, backend)

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

    def _solve_iterative(
        self,
        rhs_matrix: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        if rhs_matrix.shape[1] != 1:
            raise ValueError("gpu_iterative supports one RHS at a time")
        self._ensure_gpu_matrix()
        rhs_gpu = cp.asarray(rhs_matrix[:, 0])
        if verbose:
            print("[RUN] 阶段 2/2: 调用 GPU Jacobi-GMRES...")
        solution = self.iterative_solver.solve(
            self.A_cached_csr_gpu,
            rhs_gpu,
        )
        return cp.asnumpy(solution).reshape((-1, 1))

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
            "gpu_iterative",
            "cudss_hybrid_direct",
            "gpu_direct",
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

    def to_warp_array(self, np_field: np.ndarray) -> wp.array:
        return wp.from_numpy(
            np_field,
            dtype=wp.complex64,
            device=self.warp_engine.device,
        )
