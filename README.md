# Ultrasound Solver

这是一个三维变参数 Helmholtz 超声场求解与相位优化项目。当前核心目标是在给定障碍物、介质分布和边界条件时，求解完整三维复声场，并优化换能器阵列相位，使指定目标点或目标区域产生更强的振幅聚焦，同时抑制非目标区域旁瓣。

## 当前能力

- 三维 7 点 stencil Helmholtz 方程离散。
- 支持 Dirichlet 换能器边界、Sommerfeld 开放边界和 reflecting 刚性反射边界。
- 支持障碍物 SDF 与材料参数进入矩阵装配。
- 支持三种求解后端：
  - `gpu_iterative`
  - `gpu_direct`
  - `cudss_hybrid_direct`
- 支持边界凝聚直接法，降低大网格直接求解显存占用。
- 支持固定场景下的相位响应基：
  ```text
  u(phi) = G exp(i phi)
  ```
- 支持 `phase_only` 相位优化，以及传统几何相位 baseline 对比。
- 支持 PyVista 交互式三维可视化。

## 仓库结构

```text
src/                         源码、代码使用说明和示例配置
  README.md                  代码运行说明
  main.py                    baseline / inverse 主入口
  visualizer.py              交互式三维可视化入口
  config.py                  YAML 配置解析
  examples/                  默认配置与可复现实验配置
  solvers/                   Helmholtz 线性系统求解后端
  physics/                   阵列、SDF、Warp 矩阵/RHS 装配
  training/                  相位响应基、loss、梯度和优化器

docs/                        数学推导、求解器原理和实现路线
outputs/                     响应基、优化结果等运行产物，不提交
obstacle_displacement_study/ 障碍物轻移实验与可视化结果
requirements.txt             Python 依赖
```

## 快速入口

代码运行方式见：

[src/README.md](src/README.md)

推荐相位优化示例：

```powershell
python src/main.py --config src/examples/phase_oblique_reflecting_x_12x12.yaml
```

红蓝对比可视化：

```powershell
python src/visualizer.py outputs/phase_oblique_reflecting_x_12x12/result.npz --compare-methods --percentile 95 --hide-source-layers 4
```

## 文档索引

- `docs/main.html`：文档总入口，可跳转到各个专题页面。
- `docs/ACOUSTIC_PHYSICS_DERIVATION.html`：从流体力学到声学 Helmholtz 方程。
- `docs/HELMHOLTZ_DISCRETIZATION.html`：离散化和矩阵组装细节。
- `docs/BOUNDARY_CONDENSATION.html`：边界凝聚直接法原理。
- `docs/PHASE_ONLY_TRAINING.html`：纯相位响应基训练模式。
- `docs/GRADIENT_DERIVATION.html`：相位 VJP、振幅 loss 导数、PDE 伴随梯度和有限差分验证。
- `docs/SOLVER_ARCHITECTURE.html`：求解器架构。

## 运行产物

`outputs/`、`.npy`、`.npz` 和可视化截图属于运行产物，默认不提交。可复现实验配置应保存在 `src/examples/` 中。
