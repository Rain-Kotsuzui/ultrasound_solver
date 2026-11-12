# 16 x 16 相位优化算法对比

本目录用于在真实部署规模 `16 x 16 = 256` 阵元上比较相位优化算法。核心实现位于 `src/algorithm/baselines/` 和 `src/algorithm/compare.py`，本目录只保存可复现配置与实验产物。

## 当前配置

- 阵列：16 x 16，频率 40 kHz，阵元间距 10 mm。
- 求解域：0.16 m x 0.16 m x 0.16 m，55 x 55 x 55 网格。
- 目标点：[0.12, 0.12, 0.12] m。
- 边界：`+x` 反射，其余外边界开放，`-z` 为阵列边界。
- 损失：`focal_contrast`。
- 主效率指标：首次达到预注册共同质量门槛的时间和场评估次数。

旧的 12 x 12 结果仅为历史算法原型，不适用于当前硬件规模，也不应用于部署判断。

## 算法

| 算法 | 类型 | 可用信息 |
|---|---|---|
| `geometric` | 自由场几何相位 | 阵元与目标几何 |
| `response_alignment` | 真实响应解析对齐 | 目标点复响应 |
| `adjoint` | 解析 VJP + Adam | 响应基与相位梯度 |
| `gabs` | 贪心逐阵元搜索 | 标量损失 |
| `spsa` | 随机近似梯度 | 标量损失 |
| `cmaes` | 进化优化 | 标量损失 |
| `lshade` | L-SHADE / Differential Evolution | 标量损失 |
| `sac` | 连续动作强化学习 | 观测和奖励 |
| `ppo` | 连续动作强化学习 | 观测和奖励 |

所有迭代算法使用相同物理场景、响应基和时间预算。`adjoint` 默认使用 Adam；优化器本身的比较由 `gradient_ablation/` 负责。

## 运行

从仓库根目录执行：

```powershell
python algorithm_comparison/run_comparison.py
```

也可直接调用：

```powershell
python src/algorithm/compare.py `
  --config algorithm_comparison/config/phase_oblique_reflecting_x_16x16.yaml `
  --output-dir algorithm_comparison/results
```

默认时间预算为每种算法 60 秒；使用 `--max-seconds` 覆盖。SAC、PPO 与 CMA-ES 需要安装 `src/algorithm/baselines/requirements.txt` 中的可选依赖。

## 输出

```text
algorithm_comparison/results/
  phase_oblique_reflecting_x_16x16/
    summary.csv / summary.json
    quality_summary.csv / quality_summary.json
    loss_dashboard.png
    <algorithm>/
      config.yaml
      result.npz
      result_best_phases_rad.npy
      result_final_phases_rad.npy
      result_phase_export.json
      result_loss.png
```

`result_best_phases_rad.npy` 是推荐复用的最佳已评估连续相位。它在进入 CoreEP4CE6 前仍需经过已验证的通道映射、相位校准和 32 级量化，不能直接发送到串口。
