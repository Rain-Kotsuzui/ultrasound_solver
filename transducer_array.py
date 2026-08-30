import numpy as np
from config import SimulationConfig


class TransducerArray:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.centers = self._compute_transducer_centers()
        self.num_transducers = len(self.centers)

        # 换能器表面发射声压幅值 (统一采用标定上限 P0)
        self.surface_p0 = self.cfg.calibrated_surface_pressure
        self.amplitudes = np.full(self.num_transducers, self.surface_p0, dtype=np.float64)
        
        # 初始发射相位
        self.phases = np.zeros(self.num_transducers, dtype=np.float64)
        if self.cfg.mode == "baseline":
            self.phases = self._compute_baseline_phases()

    def _compute_transducer_centers(self) -> np.ndarray:
        """计算 N*N 阵元在底面 (z = 0) 的中心物理坐标 [x, y, 0]"""
        n = self.cfg.specs.array_n
        pitch = self.cfg.specs.pitch
        cx, cy = self.cfg.lx * 0.5, self.cfg.ly * 0.5
        half_span = (n - 1) * pitch * 0.5

        centers = []
        for i in range(n):
            x = cx - half_span + i * pitch
            for j in range(n):
                y = cy - half_span + j * pitch
                centers.append([x, y, 0.0])
        return np.array(centers, dtype=np.float64)

    def _compute_baseline_phases(self) -> np.ndarray:
        """
        Baseline 模式下的多目标时间反转加权复数场叠加:
        U_i = sum_m (1 / d_im) * exp(-i * k0 * d_im)
        phi_i = arg(U_i)
        """
        print("Computing baseline phases...")
        targets = self.cfg.targets
        if not targets:
            return np.zeros(self.num_transducers, dtype=np.float64)

        if len(targets) == 1:
            focal_pos = np.array(targets[0], dtype=np.float64)
            dists = np.linalg.norm(self.centers - focal_pos, axis=-1)
            return -self.cfg.k0 * dists

        # 多目标时间反转波前叠加
        superposed_field = np.zeros(self.num_transducers, dtype=np.complex128)
        for target_pos in targets:
            pos = np.array(target_pos, dtype=np.float64)
            dists = np.linalg.norm(self.centers - pos, axis=-1)
            superposed_field += (1.0 / dists) * np.exp(-1j * self.cfg.k0 * dists)

        return np.angle(superposed_field)

    def set_phases(self, phases: np.ndarray):
        """供逆向优化器直接更新相位的接口"""
        assert len(phases) == self.num_transducers
        self.phases = np.array(phases, dtype=np.float64)

    def get_bottom_source_boundary(self, X_2d: np.ndarray, Y_2d: np.ndarray):
        """生成底面 z=0 网格上的发射掩码与复声压 Dirichlet 边界值"""
        mask = np.zeros(X_2d.shape, dtype=bool)
        u_boundary = np.zeros(X_2d.shape, dtype=np.complex128)
        radius_sq = (self.cfg.specs.diameter * 0.5) ** 2

        for i, center in enumerate(self.centers):
            tx, ty, _ = center
            dist_sq = (X_2d - tx)**2 + (Y_2d - ty)**2
            in_transducer = dist_sq <= radius_sq
            
            mask |= in_transducer
            u_boundary[in_transducer] = self.amplitudes[i] * np.exp(1j * self.phases[i])

        return mask, u_boundary