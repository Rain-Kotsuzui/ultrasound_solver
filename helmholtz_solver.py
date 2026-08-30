import time
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as csp
import warp as wp

from nvmath.sparse.advanced import DirectSolver
from pypardiso import PyPardisoSolver
from config import SimulationConfig
from transducer_array import TransducerArray
from warp_utils import WarpAssemblyEngine


class HelmholtzDirectSolver:
    """
    智能混合直接求解器 (Smart Hybrid Direct Solver)
    - 阶段 1: Warp GPU 极速组装复数稀疏矩阵
    - 阶段 2: 智能预估显存
        - 显存充足 -> 走 NVIDIA cuDSS GPU 稀疏直接求解
        - 显存告急 -> 转为 2N x 2N 实数分块矩阵，走 CPU Pardiso 大内存多核求解
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
        print(f"[Smart Solver] 启动智能混合直接求逆 (DOFs: {self.total_dofs:,})")
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
        
        use_gpu = self._estimate_vram_and_decide_backend()
        self.solve_backend = "gpu" if use_gpu else "cpu"

        t0 = time.perf_counter()

        if self.solve_backend == "gpu":
            print("[RUN] 阶段 2/2: 调用 NVIDIA cuDSS 执行 GPU 稀疏直接分解...")
            self.A_cached_csr_gpu = csp.csr_matrix(A_csr)
            rhs_gpu = cp.array(rhs, dtype=cp.complex128)

            u_gpu = self._solve_gpu_direct(self.A_cached_csr_gpu, rhs_gpu)
            u_vec = cp.asnumpy(u_gpu)
            
        else:
            print("[RUN] 阶段 2/2: 转为 2N x 2N 纯实数矩阵并调用 Pardiso 多核分解...")
            
            A_R = A_csr.real
            A_I = A_csr.imag
            
            # 组装 2N x 2N 实数矩阵 [[Re, -Im], [Im, Re]]
            A_big = sp.bmat([
                [A_R, -A_I],
                [A_I,  A_R]
            ], format='csr')
            
            rhs_big = np.concatenate([rhs.real, rhs.imag])
            
            uv_big = self.pardiso_solver.solve(A_big, rhs_big)
            
            N = self.total_dofs
            u_vec = uv_big[:N] + 1j * uv_big[N:]
            
            self.A_cached_big_T = A_big.transpose().tocsr()

        t_solve = time.perf_counter() - t0
        print(f"[OK] 阶段 2/2: 直接求逆完成! 耗时: {t_solve:.2f} 秒")
        print("=" * 65)

        return u_vec.reshape(
            (self.cfg.domain.nx, self.cfg.domain.ny, self.cfg.domain.nz),
            order="F",
        )

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
        """伴随状态快速直接回代接口"""
        if self.solve_backend == "gpu":
            # GPU 极速回代
            A_H_gpu = self.A_cached_csr_gpu.conjugate().transpose().tocsr()
            adj_rhs_gpu = cp.array(adjoint_rhs, dtype=cp.complex128)
            v_gpu = self._solve_gpu_direct(A_H_gpu, adj_rhs_gpu)
            v_vec = cp.asnumpy(v_gpu)
            
        else:
            # CPU 极速回代 (使用缓存的 2N x 2N 实数大转置矩阵)
            adj_rhs_big = np.concatenate([adjoint_rhs.real, adjoint_rhs.imag])
            v_big = self.pardiso_solver.solve(self.A_cached_big_T, adj_rhs_big)
            
            N = self.total_dofs
            v_vec = v_big[:N] + 1j * v_big[N:]

        return v_vec.reshape(
            (self.cfg.domain.nx, self.cfg.domain.ny, self.cfg.domain.nz),
            order="F",
        )

    def to_warp_array(self, np_field: np.ndarray) -> wp.array:
        return wp.from_numpy(np_field, dtype=wp.complex64, device=self.warp_engine.device)