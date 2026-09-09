import numpy as np
from typing import List, Dict, Tuple


def smooth_heaviside(phi: np.ndarray, epsilon: float) -> np.ndarray:
    """平滑过渡带宽度为 2*epsilon"""
    H = np.zeros_like(phi, dtype=np.float64)
    in_band = np.abs(phi) <= epsilon
    outside = phi > epsilon
    
    H[outside] = 1.0
    H[in_band] = 0.5 * (1.0 + phi[in_band] / epsilon + (1.0 / np.pi) * np.sin(np.pi * phi[in_band] / epsilon))
    return H


class SDFScene:
    """基于 SDF 生成全空间连续的密度场 rho(r) 和声速场 c(r)"""
    def __init__(self, obstacles_config: List[Dict] = None):
        self.primitives = []
        if obstacles_config:
            for obs in obstacles_config:
                obs_type = obs.get("type", "").lower()
                center = np.array(obs["center"], dtype=np.float64)
                mat = obs.get("material", {"density": 1250.0, "sound_speed": 2200.0})
                
                if obs_type == "sphere":
                    self.primitives.append(('sphere', center, float(obs["radius"]), mat))
                elif obs_type == "box":
                    self.primitives.append(('box', center, np.array(obs["half_size"], dtype=np.float64), mat))

    def generate_material_fields(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                 rho0: float, c0: float, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """评估全空间网格点的物性参数"""
        p = np.stack([X, Y, Z], axis=-1)
        rho_field = np.full(X.shape, rho0, dtype=np.float64)
        c_field = np.full(X.shape, c0, dtype=np.float64)
        epsilon = 1.2 * dx  # 平滑过渡宽度

        for prim in self.primitives:
            p_type, center, geom, mat = prim[0], prim[1], prim[2], prim[3]
            if p_type == 'sphere':
                dist = np.linalg.norm(p - center, axis=-1) - geom
            elif p_type == 'box':
                d_vec = np.abs(p - center) - geom
                outside = np.linalg.norm(np.maximum(d_vec, 0.0), axis=-1)
                inside = np.minimum(np.max(d_vec, axis=-1), 0.0)
                dist = outside + inside

            # H = 1 代表空气，H = 0 代表障碍物内部
            H = smooth_heaviside(dist, epsilon)
            rho_obs, c_obs = mat["density"], mat["sound_speed"]

            rho_field = rho_obs + (rho0 - rho_obs) * H
            c_field = c_obs + (c0 - c_obs) * H

        return rho_field, c_field