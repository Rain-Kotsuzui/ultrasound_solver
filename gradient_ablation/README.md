# 解析梯度优化器消融实验

本实验固定物理场景、相位响应基、目标函数、初始相位和解析相位梯度，仅替换相位更新规则。目的不是比较不同信息条件下的算法，而是验证性能优势是否来自解析伴随梯度本身，而非仅来自 L-BFGS-B。

## 统一条件

- 场景：12 x 12 阵列、41 x 41 x 41 网格、远斜向目标点 `[0.09, 0.09, 0.09] m`、`+x` 反射边界。
- 初始相位：传统几何相位。
- 目标：`focal_contrast`。
- 梯度：所有方法均调用相同的响应基 VJP，即解析 `dL/dphi`。
- 比较：每种方法独立运行，单独计时；响应基仅构建一次，其耗时在汇总中单列。

## 更新器

| ID | 更新器 | 说明 |
|---|---|---|
| `lbfgsb` | L-BFGS-B | 二阶历史近似，现有主方法。 |
| `adam` | Adam | 自适应一阶更新。 |
| `adamw` | AdamW | 解耦权重衰减；本实验设 `weight_decay: 0`，因为相位是周期变量。 |
| `lion` | Lion | 基于符号动量的现代一阶更新。 |
| `nonlinear_cg` | Polak-Ribiere+ 非线性共轭梯度 | 采用 Armijo 回溯线搜索。 |

## 本次结果

所有方法均使用同一几何初相位，并在每个更新器 `120 s` 墙钟预算下运行。共享响应基首次构建耗时 `5.25 s`，未计入下表的单方法优化耗时。

| 更新器 | Best loss | 对比度 | 场评估次数 | VJP 次数 | 优化耗时 |
|---|---:|---:|---:|---:|---:|
| L-BFGS-B | -1593.498 | 2.559 | 306 | 306 | 120.59 s |
| Adam | -1858.794 | 2.783 | 306 | 306 | 120.70 s |
| AdamW (`weight_decay=0`) | **-1859.963** | **2.786** | 347 | 347 | 120.97 s |
| Lion | -1854.325 | 2.758 | 337 | 337 | 120.84 s |
| Nonlinear CG | -1703.420 | 2.611 | 275 | 275 | 120.74 s |

该结果说明，在这个固定场景与统一时间预算下，多个不同更新规则都能由同一解析梯度稳定降低 loss；因此性能并非只依赖某一个优化器。AdamW 此处没有施加额外正则，和 Adam 的微小差异来自不同的实际评估次数和时间边界，不应解读为权重衰减带来的物理收益。

## 结论

本消融实验表明，系统的优化优势主要来自物理模型提供的解析伴随梯度，而不是仅由某一个特定优化器造成。在统一响应基、目标函数、初始相位和 120 s 时间预算下，五种更新器均能显著降低 loss；其中 Adam 类一阶更新在本场景取得最佳结果，AdamW 的 best loss 为 `-1859.963`、对比度为 `2.786`。

L-BFGS-B、非线性共轭梯度和 Lion 同样能稳定收敛，但本轮等时间预算下不及 Adam 类更新。由于 AdamW 设置 `weight_decay: 0`，它不引入额外物理正则；与 Adam 的微小数值差异来自实际完成的场评估次数不同，不能归因于权重衰减。后续应在更多目标位置、边界和障碍物场景中重复该实验，并报告多随机种子统计。

## 指标

- `best_loss` 和 `final_loss`
- 目标区域平均振幅 `target_mean`
- 背景最大振幅 `background_max`
- 焦点/旁瓣对比度 `contrast`
- 真实声场评估次数 `field_evaluations`
- 解析 VJP 次数 `vjp_evaluations`
- 每种更新器优化耗时 `optimization_seconds`
- 共享响应基构建/加载耗时 `shared_basis_build_or_load_seconds`

`loss_dashboard.png` 显示所有更新器的 current/best loss 曲线，横轴为真实场评估次数。

## 运行

```powershell
python gradient_ablation/run_ablation.py
```

只运行指定更新器：

```powershell
python gradient_ablation/run_ablation.py --optimizers lbfgsb,adam,nonlinear_cg
```

无窗口批处理：

```powershell
python gradient_ablation/run_ablation.py --no-loss-window
```

结果写入：

```text
gradient_ablation/results/phase_oblique_reflecting_x_12x12/
  summary.csv
  summary.json
  loss_dashboard.png
  <optimizer>/
    config.yaml
    result.npz
    result_loss.png
```

首次运行会构建相位响应基缓存，后续运行自动复用。本目录不提交该缓存文件。
