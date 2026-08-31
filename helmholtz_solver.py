import time
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csplinalg
import warp as wp

from nvmath.sparse.advanced import DirectSolver
from pypardiso import PyPardisoSolver
from config import SimulationConfig
from transducer_array import TransducerArray
from warp_utils import WarpAssemblyEngine


class HelmholtzDirectSolver:
    """
    智能混合线性求解器。
    - Warp GPU 组装复数稀疏矩阵
    - 优先使用低显存的行缩放 Jacobi-GMRES
    - 迭代不收敛时回退 cuDSS GPU 或 Pardiso CPU 直接法
    """
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.transducers = TransducerArray(config)
        self.total_dofs = self.cfg.domain.nx * self.cfg.domain.ny * self.cfg.domain.nz

        self.warp_engine = WarpAssemblyEngine(
            nx=self.cfg.domain.nx,
            ny=self.cfg.domain.ny,
            nz=self.cfg.domain.nz,
            device="cuda"
        )
        
        # 状态与缓存
        self.solve_backend = "cpu"
        self.A_cached_csr_gpu = None
        self.A_cached_csr_cpu = None
        self.A_cached_big_T = None  # CPU 备用的实数大矩阵转置缓存
        
        # Pardiso 配置为 mtype=11 (实数非对称稀疏矩阵)
        self.pardiso_solver = PyPardisoSolver(mtype=11)

    def _estimate_vram_and_decide_backend(self) -> bool:
        """评估 3D 直接分解的 VRAM 需求，并决定计算后端"""
        N = self.total_dofs
        
        # 预估 L+U 填充元 (O(N^(4/3)))
        nnz_lu_estimate = 1.2 * (N ** (4.0 / 3.0))
        # Complex128 占用预估
        lu_size_gb = nnz_lu_estimate * 20.0 / (1024**3)
        # GPU 稀疏直接分解工作区预估
        workspace_gb = lu_size_gb * 7.5
        total_required_gb = lu_size_gb + workspace_gb + 0.5

        # 获取实际硬件显存
        free_mem_bytes, total_mem_bytes = cp.cuda.Device(0).mem_info
        free_mem_gb = free_mem_bytes / (1024**3)
        total_mem_gb = total_mem_bytes / (1024**3)

        print("\n" + "-" * 55)
        print("[VRAM 智能分析] 3D 直接 LU 分解资源预估:")
        print(f"  --> 空间自由度: {N:,}")
        print(f"  --> 预估 L+U 填充元: {int(nnz_lu_estimate):,} 个")
        print(f"  --> 预估分解因子显存: {lu_size_gb:.2f} GB")
        print(f"  --> 预估 GPU 分解工作区: {workspace_gb:.2f} GB")
        print(f"  --> [总计需求] 约 {total_required_gb:.2f} GB VRAM")
        print(f"  --> [硬件状态] GPU 空闲显存: {free_mem_gb:.2f} GB / {total_mem_gb:.2f} GB")
        
        is_gpu_feasible = (total_required_gb + 1.0) < free_mem_gb
        
        if is_gpu_feasible:
            print("  [OK] 结论: 显存充足，调度至 [NVIDIA cuDSS GPU 直接求解]。")
        else:
            print("  [WARN] 结论: 显存不足将触发 OOM，调度至 [Intel MKL Pardiso (系统内存)]。")
        print("-" * 55)

        return is_gpu_feasible

    def solve(self) -> np.ndarray:
        obs_cfg = self.cfg.obstacles[0] if self.cfg.obstacles else {}

        print("\n" + "=" * 65)
        print(f"[Smart Solver] 启动智能混合线性求解 (DOFs: {self.total_dofs:,})")
        print("=" * 65)

        # ----------------------------------------------------
        # 阶段 1: Warp GPU 极速矩阵装配
        # ----------------------------------------------------
        t0 = time.perf_counter()
        A_csr, rhs = self.warp_engine.assemble_system_gpu(
            dx=self.cfg.domain.dx,
            omega=self.cfg.omega,
            k0_bg=self.cfg.k0,
            rho0=self.cfg.physics.medium_density,
            c0=self.cfg.physics.sound_speed,
            obs_cfg=obs_cfg,
            trans_phases_np=self.transducers.phases,
            trans_amps_np=self.transducers.amplitudes,
            array_n=self.cfg.specs.array_n,
            trans_radius=self.cfg.specs.diameter * 0.5,
            trans_pitch=self.cfg.specs.pitch,
            bcs_dict=self.cfg.boundary_conditions
        )
        t_assemble = time.perf_counter() - t0
        print(f"[OK] 阶段 1/2: Warp GPU 矩阵装配完成 | NNZ: {A_csr.nnz:,} | 耗时: {t_assemble*1000:.2f} ms")
        
        t0 = time.perf_counter()
        self.A_cached_csr_cpu = A_csr.tocsr()
        self.A_cached_csr_gpu = csp.csr_matrix(self.A_cached_csr_cpu)
        rhs_gpu = cp.asarray(rhs, dtype=cp.complex128)

        print("[RUN] 阶段 2/2: 调用 GPU 行缩放 Jacobi-GMRES 迭代求解...")
        try:
            u_gpu = self._solve_gpu_iterative(self.A_cached_csr_gpu, rhs_gpu)
            u_vec = cp.asnumpy(u_gpu)
            self.solve_backend = "gpu_iterative"
        except RuntimeError as exc:
            print(f"  [WARN] GPU 迭代法未通过校验: {exc}")
            use_gpu_direct = self._estimate_vram_and_decide_backend()
            if use_gpu_direct:
                print("[RUN] 回退至 NVIDIA cuDSS GPU 稀疏直接分解...")
                u_gpu = self._solve_gpu_direct(self.A_cached_csr_gpu, rhs_gpu)
                u_vec = cp.asnumpy(u_gpu)
                self.solve_backend = "gpu_direct"
            else:
                print("[RUN] 回退至 2N x 2N 实数 Pardiso CPU 直接分解...")
                u_vec = self._solve_cpu_direct(self.A_cached_csr_cpu, rhs)
                self.solve_backend = "cpu"

        t_solve = time.perf_counter() - t0
        print(f"[OK] 阶段 2/2: 线性系统求解完成! 耗时: {t_solve:.2f} 秒")
        print("=" * 65)

        return u_vec.reshape(
            (self.cfg.domain.nx, self.cfg.domain.ny, self.cfg.domain.nz),
            order="F",
        )

    @staticmethod
    def _solve_gpu_iterative(A_gpu, rhs_gpu):
        if float(cp.linalg.norm(rhs_gpu).get()) == 0.0:
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
            restart=30,
            maxiter=3000,
            rtol=1.0e-6,
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
        iterations = len(residuals) * 30
        if (
            info != 0
            or not is_finite
            or scaled_residual > 1.0e-6
            or backward_error > 1.0e-4
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

    def _solve_cpu_direct(self, A_csr, rhs):
        A_big = sp.bmat(
            [[A_csr.real, -A_csr.imag], [A_csr.imag, A_csr.real]],
            format="csr",
        )
        rhs_big = np.concatenate([rhs.real, rhs.imag])
        uv_big = self.pardiso_solver.solve(A_big, rhs_big)
        self.A_cached_big_T = A_big.transpose().tocsr()
        return uv_big[:self.total_dofs] + 1j * uv_big[self.total_dofs:]

    @staticmethod
    def _solve_gpu_direct(A_gpu, rhs_gpu):
        with DirectSolver(A_gpu, rhs_gpu) as solver:
            solver.plan()
            solver.factorize()
            solution = solver.solve()

        relative_residual = cp.linalg.norm(A_gpu @ solution - rhs_gpu) / cp.linalg.norm(rhs_gpu)
        relative_residual = float(relative_residual.get())
        is_finite = bool(cp.isfinite(solution).all().get())
        if not is_finite or relative_residual > 1.0e-4:
            raise RuntimeError(
                f"GPU direct solve failed validation: relative residual={relative_residual:.3e}"
            )
        print(f"  --> cuDSS 相对残差: {relative_residual:.3e}")
        return solution

    def solve_adjoint(self, adjoint_rhs: np.ndarray) -> np.ndarray:
        """求解 A^H v = adjoint_rhs；优先使用与正向一致的 GPU 迭代法。"""
        adjoint_rhs_vec = np.asarray(adjoint_rhs, dtype=np.complex128).reshape(
            -1, order="F"
        )
        if adjoint_rhs_vec.size != self.total_dofs:
            raise ValueError(
                f"adjoint_rhs has {adjoint_rhs_vec.size} entries, "
                f"expected {self.total_dofs}"
            )

        if self.solve_backend in {"gpu_iterative", "gpu_direct"}:
            A_H_cpu = self.A_cached_csr_cpu.conjugate().transpose().tocsr()
            A_H_gpu = csp.csr_matrix(A_H_cpu)
            adj_rhs_gpu = cp.asarray(adjoint_rhs_vec)
            if self.solve_backend == "gpu_iterative":
                try:
                    v_gpu = self._solve_gpu_iterative(A_H_gpu, adj_rhs_gpu)
                    v_vec = cp.asnumpy(v_gpu)
                except RuntimeError as exc:
                    print(f"  [WARN] GPU 伴随迭代未通过校验: {exc}")
                    if self._estimate_vram_and_decide_backend():
                        print("[RUN] 伴随求解回退至 NVIDIA cuDSS...")
                        v_gpu = self._solve_gpu_direct(A_H_gpu, adj_rhs_gpu)
                        v_vec = cp.asnumpy(v_gpu)
                    else:
                        print("[RUN] 伴随求解回退至 Pardiso CPU 直接法...")
                        v_vec = self._solve_cpu_direct(A_H_cpu, adjoint_rhs_vec)
            else:
                v_gpu = self._solve_gpu_direct(A_H_gpu, adj_rhs_gpu)
                v_vec = cp.asnumpy(v_gpu)

        else:
            adj_rhs_big = np.concatenate(
                [adjoint_rhs_vec.real, adjoint_rhs_vec.imag]
            )
            v_big = self.pardiso_solver.solve(self.A_cached_big_T, adj_rhs_big)
            v_vec = v_big[:self.total_dofs] + 1j * v_big[self.total_dofs:]

        return v_vec.reshape(
            (self.cfg.domain.nx, self.cfg.domain.ny, self.cfg.domain.nz),
            order="F",
        )

    def to_warp_array(self, np_field: np.ndarray) -> wp.array:
        return wp.from_numpy(np_field, dtype=wp.complex64, device=self.warp_engine.device)
