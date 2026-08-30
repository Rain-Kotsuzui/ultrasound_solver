import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class PhysicsConfig:
    sound_speed: float = 343.0
    medium_density: float = 1.21


@dataclass
class TransducerSpecsConfig:
    frequency: float = 40000.0     # Hz
    spl_db: float = 115.0          # dB
    spl_distance: float = 0.3      # m
    diameter: float = 0.0098       # m
    pitch: float = 0.0105          # m
    array_n: int = 6


@dataclass
class DomainConfig:
    # 物理总边长 [Lx, Ly, Lz] (m)
    box_size: List[float] = field(default_factory=lambda: [0.08, 0.08, 0.08])
    # 离散网格点数 [Nx, Ny, Nz]
    grid_size: List[int] = field(default_factory=lambda: [81, 81, 81])

    @property
    def nx(self) -> int: return self.grid_size[0]
    @property
    def ny(self) -> int: return self.grid_size[1]
    @property
    def nz(self) -> int: return self.grid_size[2]

    @property
    def lx(self) -> float: return self.box_size[0]
    @property
    def ly(self) -> float: return self.box_size[1]
    @property
    def lz(self) -> float: return self.box_size[2]

    @property
    def dx(self) -> float:
        """ 
        dx = Lx / (Nx - 1)
        """
        calc_dx = self.lx / (self.nx - 1)
        calc_dy = self.ly / (self.ny - 1)
        calc_dz = self.lz / (self.nz - 1)

        # 是否为正方各向同性网格
        if not (np.isclose(calc_dx, calc_dy) and np.isclose(calc_dx, calc_dz)):
            raise ValueError(
                f"当前求解器要求各向同性等步长网格 (dx=dy=dz)，但计算得到: "
                f"dx={calc_dx*1000:.3f}mm, dy={calc_dy*1000:.3f}mm, dz={calc_dz*1000:.3f}mm。"
                f"请检查 box_size 与 grid_size 的比例是否一致！"
            )
        return float(calc_dx)


@dataclass
class IOConfig:
    output_file: str = "ultrasound_field.npz"
    auto_visualize: bool = True
    colormap: str = "plasma"


@dataclass
class SimulationConfig:
    mode: str
    physics: PhysicsConfig
    specs: TransducerSpecsConfig
    domain: DomainConfig
    boundary_conditions: Dict[str, str]
    targets: List[List[float]]
    obstacles: List[Dict]
    io: IOConfig

    @property
    def frequency(self) -> float:
        return self.specs.frequency

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.specs.frequency

    @property
    def k0(self) -> float:
        return self.omega / self.physics.sound_speed

    @property
    def wavelength(self) -> float:
        return self.physics.sound_speed / self.specs.frequency

    @property
    def lx(self) -> float: return self.domain.lx
    @property
    def ly(self) -> float: return self.domain.ly
    @property
    def lz(self) -> float: return self.domain.lz

    @property
    def calibrated_surface_pressure(self) -> float:
        """基于圆形活塞声辐射模型，从 SPL 标定反推发射表面声压 P0 (Pa)"""
        p_ref = 20e-6
        p_rms = p_ref * (10.0 ** (self.specs.spl_db / 20.0))
        p_peak = np.sqrt(2.0) * p_rms
        radius = self.specs.diameter * 0.5
        
        p0_surface = p_peak * (2.0 * self.specs.spl_distance) / (self.k0 * (radius ** 2))
        return float(p0_surface)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SimulationConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        targets_list = [t["point"] for t in raw.get("targets", [])]

        return cls(
            mode=raw.get("mode", "baseline").lower(),
            physics=PhysicsConfig(**raw["physics"]),
            specs=TransducerSpecsConfig(**raw["transducer_specs"]),
            domain=DomainConfig(**raw["domain"]),
            boundary_conditions=raw["boundary_conditions"],
            targets=targets_list,
            obstacles=raw.get("obstacles", []),
            io=IOConfig(**raw.get("io", {}))
        )