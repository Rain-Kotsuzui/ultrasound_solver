# 代码使用说明


## 运行

使用默认配置：

```powershell
python src/main.py
```

指定配置文件：

```powershell
python src/main.py --config src/examples/config.yaml
```

运行远斜向目标点、`+x` 反射边界、传统方法对比示例：

```powershell
python src/main.py --config src/examples/phase_oblique_reflecting_x_12x12.yaml
```

## 可视化

列出结果文件中的可视化字段：

```powershell
python src/visualizer.py outputs/phase_oblique_reflecting_x_12x12/result.npz --list-fields
```

查看优化后的振幅场：

```powershell
python src/visualizer.py outputs/phase_oblique_reflecting_x_12x12/result.npz --field amplitude --percentile 95 --hide-source-layers 4
```

同屏对比传统几何相位和优化相位：

```powershell
python src/visualizer.py outputs/phase_oblique_reflecting_x_12x12/result.npz --compare-methods --percentile 95 --hide-source-layers 4
```

红色等值面表示传统几何相位结果，蓝色等值面表示优化相位结果；两个滑条分别控制两种方法的等值面阈值。

## Config 参数

### 顶层

- `mode`: 运行模式，`baseline` 表示前向求解，`inverse` 表示进入逆向/优化流程。

### `physics`

- `sound_speed`: 背景介质声速，单位 `m/s`。
- `medium_density`: 背景介质密度，单位 `kg/m^3`。

### `transducer_specs`

- `frequency`: 换能器频率，单位 `Hz`。
- `spl_db`: 换能器声压级标定值，单位 `dB`。
- `spl_distance`: 声压级标定距离，单位 `m`。
- `diameter`: 单个换能器有效直径，单位 `m`。
- `pitch`: 阵元中心间距，单位 `m`。
- `array_n`: 方形阵列单边阵元数量，总阵元数为 `array_n * array_n`。

### `domain`

- `box_size`: 求解域尺寸 `[Lx, Ly, Lz]`，单位 `m`。
- `grid_size`: 网格点数 `[Nx, Ny, Nz]`。当前求解器要求三方向网格步长一致。

### `solver`

- `backend`: 求解后端，支持 `gpu_iterative`、`gpu_direct`、`cudss_hybrid_direct`。
- `direct_residual_tol`: 直接法相对残差检查阈值。
- `hybrid_num_threads`: cuDSS hybrid CPU 线程数。
- `hybrid_execution_mode`: cuDSS hybrid 模式，支持 `auto`、`hybrid_execute`、`hybrid_memory`。
- `hybrid_device_memory_limit_gb`: cuDSS hybrid 可使用的 GPU 显存上限，单位 `GB`。
- `hybrid_device_memory_reserve_gb`: 为其他 CUDA/Warp 用途预留的显存，单位 `GB`。
- `hybrid_register_cuda_memory`: 是否注册 CUDA 主存。
- `condensed_symmetry_tolerance`: 边界凝聚约化矩阵的复对称性检查阈值。
- `gmres_restart`: GMRES 每轮 Krylov 子空间大小。
- `gmres_maxiter`: GMRES 最大重启轮数。
- `gmres_rtol`: GMRES 相对残差阈值。
- `gmres_backward_error_tol`: 原系统后向误差阈值。
- `reuse_matrix`: 仅相位变化时是否复用矩阵装配结果。

### `training`

- `mode`: 训练模式，当前主线使用 `phase_only`。
- `loss_type`: 损失函数，支持 `field_match`、`focal_pressure`、`focal_contrast`。
- `initial_phase`: 初始相位，支持 `baseline`、`response`、`current`、`zero`。
- `compare_baseline`: 是否额外计算并保存传统几何相位结果，用于红蓝对比可视化。
- `phase_basis_file`: 相位响应基缓存文件路径。
- `phase_basis_batch_size`: 构建响应基时每批处理的阵元数量。
- `voxel_chunk_size`: CPU 合成场和反向梯度计算的体素分块大小。
- `load_basis_to_gpu`: 是否将响应基常驻 GPU。
- `release_factor_after_basis`: 构建响应基后是否释放直接求解器因子。
- `target_field_file`: 外部目标振幅场文件；为空时根据 `targets` 自动生成高斯目标场。
- `target_peak_pressure`: 目标点峰值振幅，单位 `Pa`。
- `target_sigma`: 高斯目标场标准差，单位 `m`。
- `background_pressure`: 非目标区域期望振幅，单位 `Pa`。
- `target_weight`: 目标区域权重。
- `background_weight`: 背景/旁瓣抑制权重。
- `target_radius`: `focal_contrast` 中从旁瓣区域排除的目标半径，单位 `m`。
- `source_exclusion_layers`: 忽略底部源面附近的网格层数。
- `sidelobe_temperature`: `focal_contrast` 平滑最大旁瓣的温度参数，单位 `Pa`。
- `optimizer`: 相位优化器，支持 `adam`、`lbfgsb`。
- `learning_rate`: Adam 学习率。
- `iterations`: 优化迭代次数。
- `gradient_check`: 是否在优化前执行有限差分梯度检查。
- `gradient_check_step`: 有限差分相位步长，单位 `rad`。

### `boundary_conditions`

- `-x`、`+x`、`-y`、`+y`、`-z`、`+z`: 六个边界面的边界条件，支持 `open` 和 `reflecting`。底面 `-z` 中的换能器区域由阵列 Dirichlet 边界控制。

### `targets`

- `point`: 目标点物理坐标 `[x, y, z]`，单位 `m`。可配置多个目标点。

### `obstacles`

- `type`: 障碍物类型，支持 `sphere`、`mesh`。
- `center` / `center_m`: 障碍物中心或变换后包围盒中心，单位 `m`。
- `radius`: 球体半径，单位 `m`。
- `file`: mesh 文件路径。
- `rotation_deg`: mesh 绕 `X/Y/Z` 轴旋转角，单位 `deg`。
- `scale`: mesh 原始单位到 `m` 的缩放比例。
- `smoothing_width_cells`: SDF 平滑界面半宽，单位为网格单元数。
- `material.density`: 障碍物密度，单位 `kg/m^3`。
- `material.sound_speed`: 障碍物声速，单位 `m/s`。

### `io`

- `output_file`: 结果 `.npz` 输出路径。
- `auto_visualize`: 求解后是否自动打开可视化。
- `colormap`: 振幅场可视化颜色表。
