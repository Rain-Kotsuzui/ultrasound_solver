import warp as wp
import numpy as np
import scipy.sparse as sp

wp.init()

@wp.func
def sdf_sphere(p: wp.vec3, center: wp.vec3, radius: float) -> float:
    return wp.length(p - center) - radius

@wp.func
def smooth_heaviside(phi: float, eps: float) -> float:
    if phi < -eps:
        return float(0.0)
    elif phi > eps:
        return float(1.0)
    else:
        return 0.5 * (1.0 + phi / eps + (1.0 / wp.pi) * wp.sin(wp.pi * phi / eps))

# -------------------------------------------------------------------------
# Warp Kernel 1: 全空间连续介质物理场生成 (rho, c)
# -------------------------------------------------------------------------
@wp.kernel
def compute_material_fields_kernel(
    rho_field: wp.array(dtype=float, ndim=3),
    c_field: wp.array(dtype=float, ndim=3),
    nx: int, ny: int, nz: int, dx: float,
    rho0: float, c0: float,
    rho_obs: float, c_obs: float,
    obs_center: wp.vec3, obs_radius: float
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    p = wp.vec3(float(i) * dx, float(j) * dx, float(k) * dx)
    phi = sdf_sphere(p, obs_center, obs_radius)
    H = smooth_heaviside(phi, 1.2 * dx)

    rho_field[i, j, k] = rho_obs + (rho0 - rho_obs) * H
    c_field[i, j, k] = c_obs + (c0 - c_obs) * H

# -------------------------------------------------------------------------
# Warp Kernel 2: 核心 3D 变参数 Helmholtz 极速装配 Kernel (COO 格式)
# -------------------------------------------------------------------------
@wp.kernel
def assemble_helmholtz_kernel(
    # 输出 COO 缓冲区 (大小为 7 * N_total)
    coo_rows: wp.array(dtype=int),
    coo_cols: wp.array(dtype=int),
    coo_vals_r: wp.array(dtype=float),
    coo_vals_i: wp.array(dtype=float),
    rhs_r: wp.array(dtype=float),
    rhs_i: wp.array(dtype=float),
    # 预计算物理场
    rho_field: wp.array(dtype=float, ndim=3),
    c_field: wp.array(dtype=float, ndim=3),
    # 发射阵列参数
    trans_phases: wp.array(dtype=float),
    trans_amps: wp.array(dtype=float),
    array_n: int, trans_radius: float, trans_pitch: float,
    # 几何与物理常数
    nx: int, ny: int, nz: int, dx: float,
    omega: float, k0_bg: float,
    # 6 个边界条件标志位 (0: reflecting, 1: open)
    bc_xm: int, bc_xp: int,
    bc_ym: int, bc_yp: int,
    bc_zm: int, bc_zp: int
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    idx = i + j * nx + k * (nx * ny)
    base_slot = idx * 7  # 每个网格节点分配 7 个非零元位置
    inv_dx = 1.0 / dx
    inv_dx2 = 1.0 / (dx * dx)
    p = wp.vec3(float(i) * dx, float(j) * dx, float(k) * dx)

    # 1. 检查底面 (z = 0) 换能器圆盘区域 (使用 O(1) 最近邻几何映射，消除动态循环)
    is_emitter = int(0)
    emitter_idx = int(0)
    
    if k == 0:
        center_x = float(nx) * dx * 0.5
        center_y = float(ny) * dx * 0.5
        half_span = float(array_n - 1) * trans_pitch * 0.5
        start_x = center_x - half_span
        start_y = center_y - half_span

        # 计算最近的阵元行列索引 (ax, ay)
        ax = int(wp.round((p[0] - start_x) / trans_pitch))
        ay = int(wp.round((p[1] - start_y) / trans_pitch))

        if ax >= 0 and ax < array_n and ay >= 0 and ay < array_n:
            tx = start_x + float(ax) * trans_pitch
            ty = start_y + float(ay) * trans_pitch
            dist_sq = (p[0] - tx) * (p[0] - tx) + (p[1] - ty) * (p[1] - ty)
            if dist_sq <= trans_radius * trans_radius:
                is_emitter = int(1)
                emitter_idx = ax * array_n + ay

    # ----------------------------------------------------
    # 分支 A: 换能器激励点 (Dirichlet: u = A * exp(i * phi))
    # ----------------------------------------------------
    if is_emitter == 1:
        coo_rows[base_slot] = idx
        coo_cols[base_slot] = idx
        coo_vals_r[base_slot] = 1.0
        coo_vals_i[base_slot] = 0.0

        amp = trans_amps[emitter_idx]
        ph = trans_phases[emitter_idx]
        rhs_r[idx] = amp * wp.cos(ph)
        rhs_i[idx] = amp * wp.sin(ph)

        # 其余 6 个邻居槽位填 0
        for s in range(1, 7):
            coo_rows[base_slot + s] = idx
            coo_cols[base_slot + s] = idx
            coo_vals_r[base_slot + s] = 0.0
            coo_vals_i[base_slot + s] = 0.0
        return

    # ----------------------------------------------------
    # 分支 B: 开放吸收边界点 (Sommerfeld ABC: ∂u/∂n - i*k0*u = 0)
    # ----------------------------------------------------
    is_open_boundary = int(0)
    nbr_idx_abc = int(0)

    if i == 0 and bc_xm == 1:
        is_open_boundary = int(1); nbr_idx_abc = (i + 1) + j * nx + k * (nx * ny)
    elif i == nx - 1 and bc_xp == 1:
        is_open_boundary = int(1); nbr_idx_abc = (i - 1) + j * nx + k * (nx * ny)
    elif j == 0 and bc_ym == 1:
        is_open_boundary = int(1); nbr_idx_abc = i + (j + 1) * nx + k * (nx * ny)
    elif j == ny - 1 and bc_yp == 1:
        is_open_boundary = int(1); nbr_idx_abc = i + (j - 1) * nx + k * (nx * ny)
    elif k == 0 and bc_zm == 1:
        is_open_boundary = int(1); nbr_idx_abc = i + j * nx + (k + 1) * (nx * ny)
    elif k == nz - 1 and bc_zp == 1:
        is_open_boundary = int(1); nbr_idx_abc = i + j * nx + (k - 1) * (nx * ny)

    if is_open_boundary == 1:
        coo_rows[base_slot] = idx
        coo_cols[base_slot] = idx
        coo_vals_r[base_slot] = inv_dx
        coo_vals_i[base_slot] = -k0_bg

        coo_rows[base_slot + 1] = idx
        coo_cols[base_slot + 1] = nbr_idx_abc
        coo_vals_r[base_slot + 1] = -inv_dx
        coo_vals_i[base_slot + 1] = 0.0

        for s in range(2, 7):
            coo_rows[base_slot + s] = idx
            coo_cols[base_slot + s] = idx
            coo_vals_r[base_slot + s] = 0.0
            coo_vals_i[base_slot + s] = 0.0
        return

    # ----------------------------------------------------
    # 分支 C: 内部及全反射壁面点 (变参数 7 点半网格密度差分)
    # ----------------------------------------------------
    rho_c = rho_field[i, j, k]
    c_c = c_field[i, j, k]

    diag_r = (omega * omega) / (rho_c * c_c * c_c)
    sum_flux = float(0.0)

    # Slot 1: -x
    if i > 0:
        rho_n = rho_field[i - 1, j, k]
        c_val = inv_dx2 * (2.0 / (rho_c + rho_n))
        coo_rows[base_slot + 1] = idx
        coo_cols[base_slot + 1] = (i - 1) + j * nx + k * (nx * ny)
        coo_vals_r[base_slot + 1] = c_val
        coo_vals_i[base_slot + 1] = 0.0
        sum_flux += c_val
    else:
        coo_rows[base_slot + 1] = idx; coo_cols[base_slot + 1] = idx
        coo_vals_r[base_slot + 1] = 0.0; coo_vals_i[base_slot + 1] = 0.0

    # Slot 2: +x
    if i < nx - 1:
        rho_n = rho_field[i + 1, j, k]
        c_val = inv_dx2 * (2.0 / (rho_c + rho_n))
        coo_rows[base_slot + 2] = idx
        coo_cols[base_slot + 2] = (i + 1) + j * nx + k * (nx * ny)
        coo_vals_r[base_slot + 2] = c_val
        coo_vals_i[base_slot + 2] = 0.0
        sum_flux += c_val
    else:
        coo_rows[base_slot + 2] = idx; coo_cols[base_slot + 2] = idx
        coo_vals_r[base_slot + 2] = 0.0; coo_vals_i[base_slot + 2] = 0.0

    # Slot 3: -y
    if j > 0:
        rho_n = rho_field[i, j - 1, k]
        c_val = inv_dx2 * (2.0 / (rho_c + rho_n))
        coo_rows[base_slot + 3] = idx
        coo_cols[base_slot + 3] = i + (j - 1) * nx + k * (nx * ny)
        coo_vals_r[base_slot + 3] = c_val
        coo_vals_i[base_slot + 3] = 0.0
        sum_flux += c_val
    else:
        coo_rows[base_slot + 3] = idx; coo_cols[base_slot + 3] = idx
        coo_vals_r[base_slot + 3] = 0.0; coo_vals_i[base_slot + 3] = 0.0

    # Slot 4: +y
    if j < ny - 1:
        rho_n = rho_field[i, j + 1, k]
        c_val = inv_dx2 * (2.0 / (rho_c + rho_n))
        coo_rows[base_slot + 4] = idx
        coo_cols[base_slot + 4] = i + (j + 1) * nx + k * (nx * ny)
        coo_vals_r[base_slot + 4] = c_val
        coo_vals_i[base_slot + 4] = 0.0
        sum_flux += c_val
    else:
        coo_rows[base_slot + 4] = idx; coo_cols[base_slot + 4] = idx
        coo_vals_r[base_slot + 4] = 0.0; coo_vals_i[base_slot + 4] = 0.0

    # Slot 5: -z
    if k > 0:
        rho_n = rho_field[i, j, k - 1]
        c_val = inv_dx2 * (2.0 / (rho_c + rho_n))
        coo_rows[base_slot + 5] = idx
        coo_cols[base_slot + 5] = i + j * nx + (k - 1) * (nx * ny)
        coo_vals_r[base_slot + 5] = c_val
        coo_vals_i[base_slot + 5] = 0.0
        sum_flux += c_val
    else:
        coo_rows[base_slot + 5] = idx; coo_cols[base_slot + 5] = idx
        coo_vals_r[base_slot + 5] = 0.0; coo_vals_i[base_slot + 5] = 0.0

    # Slot 6: +z
    if k < nz - 1:
        rho_n = rho_field[i, j, k + 1]
        c_val = inv_dx2 * (2.0 / (rho_c + rho_n))
        coo_rows[base_slot + 6] = idx
        coo_cols[base_slot + 6] = i + j * nx + (k + 1) * (nx * ny)
        coo_vals_r[base_slot + 6] = c_val
        coo_vals_i[base_slot + 6] = 0.0
        sum_flux += c_val
    else:
        coo_rows[base_slot + 6] = idx; coo_cols[base_slot + 6] = idx
        coo_vals_r[base_slot + 6] = 0.0; coo_vals_i[base_slot + 6] = 0.0

    # Slot 0: 对角线
    coo_rows[base_slot] = idx
    coo_cols[base_slot] = idx
    coo_vals_r[base_slot] = diag_r - sum_flux
    coo_vals_i[base_slot] = 0.0


# -------------------------------------------------------------------------
# Python 引擎封装类
# -------------------------------------------------------------------------
class WarpAssemblyEngine:
    def __init__(self, nx: int, ny: int, nz: int, device: str = "cuda"):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.total_dofs = nx * ny * nz
        self.total_entries = self.total_dofs * 7
        self.device = device if wp.is_cuda_available() and device == "cuda" else "cpu"

        # 预分配连续 GPU 显存池
        self.rho_field = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.c_field = wp.zeros((nx, ny, nz), dtype=float, device=self.device)

        self.coo_rows = wp.zeros(self.total_entries, dtype=int, device=self.device)
        self.coo_cols = wp.zeros(self.total_entries, dtype=int, device=self.device)
        self.coo_vals_r = wp.zeros(self.total_entries, dtype=float, device=self.device)
        self.coo_vals_i = wp.zeros(self.total_entries, dtype=float, device=self.device)
        self.rhs_r = wp.zeros(self.total_dofs, dtype=float, device=self.device)
        self.rhs_i = wp.zeros(self.total_dofs, dtype=float, device=self.device)

    def assemble_system_gpu(
        self,
        dx: float, omega: float, k0_bg: float,
        rho0: float, c0: float,
        obs_cfg: dict,
        trans_phases_np: np.ndarray,
        trans_amps_np: np.ndarray,
        array_n: int, trans_radius: float, trans_pitch: float,
        bcs_dict: dict
    ) -> tuple:
        """调用 GPU Kernel 极速装配稀疏系统"""
        # 1. 障碍物参数
        obs_center = wp.vec3(*obs_cfg.get("center", [0.04, 0.04, 0.02]))
        obs_radius = float(obs_cfg.get("radius", 0.012))
        mat = obs_cfg.get("material", {})
        rho_obs = float(mat.get("density", 1250.0))
        c_obs = float(mat.get("sound_speed", 2200.0))

        # 2. 生成连续介质物性场
        wp.launch(
            kernel=compute_material_fields_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self.rho_field, self.c_field,
                self.nx, self.ny, self.nz, dx,
                rho0, c0, rho_obs, c_obs,
                obs_center, obs_radius
            ],
            device=self.device
        )

        # 3. 边界条件标志位 (0: reflecting, 1: open)
        bc_flags = [1 if bcs_dict.get(k) == "open" else 0 for k in ["-x", "+x", "-y", "+y", "-z", "+z"]]

        # 4. 上传阵列相位与幅值
        trans_phases_wp = wp.from_numpy(trans_phases_np.astype(np.float32), dtype=float, device=self.device)
        trans_amps_wp = wp.from_numpy(trans_amps_np.astype(np.float32), dtype=float, device=self.device)

        # 5. 启动 GPU 矩阵装配 Kernel
        wp.launch(
            kernel=assemble_helmholtz_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self.coo_rows, self.coo_cols,
                self.coo_vals_r, self.coo_vals_i,
                self.rhs_r, self.rhs_i,
                self.rho_field, self.c_field,
                trans_phases_wp, trans_amps_wp,
                array_n, trans_radius, trans_pitch,
                self.nx, self.ny, self.nz, dx,
                omega, k0_bg,
                bc_flags[0], bc_flags[1],
                bc_flags[2], bc_flags[3],
                bc_flags[4], bc_flags[5]
            ],
            device=self.device
        )

        # 6. 转为 CSC 格式
        rows = self.coo_rows.numpy()
        cols = self.coo_cols.numpy()
        data = self.coo_vals_r.numpy() + 1j * self.coo_vals_i.numpy()
        rhs = self.rhs_r.numpy() + 1j * self.rhs_i.numpy()

        valid_mask = (data != 0.0)
        A = sp.csc_matrix((data[valid_mask], (rows[valid_mask], cols[valid_mask])),
                          shape=(self.total_dofs, self.total_dofs), dtype=np.complex128)

        return A, rhs