# 16 x 16 解析梯度优化器消融实验

本实验固定 16 x 16 物理场景、相位响应基、目标函数、初始相位和解析相位梯度，仅替换相位更新规则。目标是验证性能收益来自解析伴随梯度，而不是某个单独优化器。

## 统一条件

- 阵列：16 x 16，40 kHz，256 个相位变量。
- 求解域：0.16 m x 0.16 m x 0.16 m，55 x 55 x 55 网格。
- 目标点：[0.12, 0.12, 0.12] m，`+x` 反射边界。
- 初始相位：几何相位。
- 损失：`focal_contrast`。
- 梯度：所有方法调用同一响应基 VJP。
- 停止协议：统一 60 秒墙钟上限与 15000 次场评估保护上限。

旧的 12 x 12 数据不代表当前部署规模，不能用于 16 x 16 的效率或效果结论。新的结果会在本配置重新运行后写入 `results/phase_oblique_reflecting_x_16x16/`。

## 更新器

| ID | 更新器 |
|---|---|
| `lbfgsb` | L-BFGS-B |
| `adam` | Adam |
| `adamw` | AdamW，默认 `weight_decay: 0` |
| `lion` | Lion |
| `nonlinear_cg` | Polak-Ribiere+ 非线性共轭梯度 |

## 运行

```powershell
python gradient_ablation/run_ablation.py
```

指定部分更新器：

```powershell
python gradient_ablation/run_ablation.py `
  --optimizers lbfgsb,adam,nonlinear_cg
```

无窗口批处理：

```powershell
python gradient_ablation/run_ablation.py --no-loss-window
```

## 输出

```text
gradient_ablation/results/
  phase_oblique_reflecting_x_16x16/
    summary.csv / summary.json
    quality_summary.csv / quality_summary.json
    loss_dashboard.png
    <optimizer>/
      config.yaml
      result.npz
      result_best_phases_rad.npy
      result_final_phases_rad.npy
      result_phase_export.json
      result_loss.png
```

`result_best_phases_rad.npy` 是推荐复用的最佳已评估连续相位。发送到 CoreEP4CE6 前必须由硬件服务完成通道映射、校准与 32 级量化。
