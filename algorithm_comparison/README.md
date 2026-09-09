# 相位优化算法对比

本目录是可上传的算法对比实验包，包含固定超声场景的完整配置、8 种算法的运行产物和可视化曲线。核心实现位于仓库的 `src/baselines/` 与 `src/compare.py`，本目录不复制算法源码，避免产生两份不一致的实现。

## 场景

- 阵列：12 x 12，频率 40 kHz。
- 求解域：0.12 m x 0.12 m x 0.12 m，41 x 41 x 41 网格。
- 目标点：[0.09, 0.09, 0.09] m。
- 边界：`+x` 为反射边界，其余外边界开放，`-z` 为换能器阵列边界。
- 优化目标：`focal_contrast`，在目标区域增强振幅，同时压制非目标区域旁瓣。
- 对比指标：best loss、目标区域平均振幅、背景最大振幅、焦点/旁瓣对比度、场评估次数和优化耗时。

## 算法

| 算法 | 类型 | 已知信息 |
|---|---|---|
| `geometric` | 自由场几何相位 | 阵元和目标几何 |
| `response_alignment` | 真实响应解析对齐 | 目标点复响应 |
| `adjoint` | 本方法，解析 VJP + L-BFGS-B | 响应基与相位梯度 |
| `gabs` | 贪心逐阵元离散相位搜索 | 标量损失 |
| `spsa` | 随机近似梯度 | 标量损失 |
| `cmaes` | 进化优化 | 标量损失 |
| `sac` | 连续动作强化学习 | 观测和奖励 |
| `ppo` | 连续动作强化学习 | 观测和奖励 |

`geometric` 和 `response_alignment` 是解析基线，不进行迭代。其余方法从同一几何相位初始化，并受统一的 60 s 墙钟预算和各自的场评估上限约束。

## 本次结果

| 算法 | Best loss | 对比度 | 场评估次数 | 优化耗时 |
|---|---:|---:|---:|---:|
| geometric | 1788.425 | 0.306 | 1 | 1.19 s |
| response_alignment | -1220.675 | 1.883 | 1 | 0.69 s |
| adjoint | **-1593.498** | **2.559** | 323 | 85.31 s |
| gabs | -1141.359 | 2.090 | 2305 | 44.67 s |
| spsa | 733.987 | 0.747 | 2401 | 125.02 s |
| cmaes | -1133.107 | 2.060 | 2401 | 65.62 s |
| sac | 957.413 | 0.600 | 8284 | 278.82 s |
| ppo | 1000.664 | 0.555 | 8482 | 149.88 s |

在此固定场景中，伴随梯度 + L-BFGS-B 取得最低 loss 和最高焦点/旁瓣对比度。RL 从零开始训练的总成本被完整计入，不应与预训练后单次推理的成本混淆。

## 目录

```text
algorithm_comparison/
  config/
    phase_oblique_reflecting_x_12x12.yaml
  results/
    phase_oblique_reflecting_x_12x12/
      summary.csv / summary.json
      loss_dashboard.png
      <algorithm>/
        config.yaml
        result.npz
        loss_curve.png
        policy.zip / policy.json       # 仅 SAC、PPO
      tensorboard/
        events.out.tfevents.*
  run_comparison.py
```

`result.npz` 包含最终相位、最佳已评估相位、复声压场、振幅场、目标场、几何基线场、loss 历史和运行元数据。`summary.csv` 是便于表格软件读取的汇总，`summary.json` 保留相同的结构化信息。

## 复现

从仓库根目录运行：

```powershell
python algorithm_comparison/run_comparison.py
```

该命令将重新运行所有算法，并覆盖 `algorithm_comparison/results/phase_oblique_reflecting_x_12x12/`。上传包不包含大型相位响应基缓存，首次运行会自动构建，后续运行自动复用本地缓存。需要安装项目基础依赖；SAC、PPO 与 CMA-ES 还需要 `src/baselines/requirements.txt` 中的可选依赖。

查看本次记录的交互式曲线：

```powershell
tensorboard --logdir algorithm_comparison/results/phase_oblique_reflecting_x_12x12/tensorboard
```

所有图的横轴均为真实场评估次数。蓝线是当前候选的 loss，红线是截至该评估次数的最优 loss。
