# 相位优化算法对比

本目录是可上传的算法对比实验包，包含固定超声场景的完整配置、9 种算法的运行产物和可视化曲线。核心实现位于仓库的 `src/baselines/` 与 `src/compare.py`，本目录不复制算法源码，避免产生两份不一致的实现。

## 场景

- 阵列：12 x 12，频率 40 kHz。
- 求解域：0.12 m x 0.12 m x 0.12 m，41 x 41 x 41 网格。
- 目标点：[0.09, 0.09, 0.09] m。
- 边界：`+x` 为反射边界，其余外边界开放，`-z` 为换能器阵列边界。
- 优化目标：`focal_contrast`，在目标区域增强振幅，同时压制非目标区域旁瓣。
- 对比指标：best loss、目标区域平均振幅、背景最大振幅、焦点/旁瓣对比度、场评估次数，以及首次达到共同质量门槛的时间。

## 算法

| 算法 | 类型 | 已知信息 |
|---|---|---|
| `geometric` | 自由场几何相位 | 阵元和目标几何 |
| `response_alignment` | 真实响应解析对齐 | 目标点复响应 |
| `adjoint` | 本方法，解析 VJP + L-BFGS-B | 响应基与相位梯度 |
| `gabs` | 贪心逐阵元离散相位搜索 | 标量损失 |
| `spsa` | 随机近似梯度 | 标量损失 |
| `cmaes` | 进化优化 | 标量损失 |
| `lshade` | L-SHADE / Differential Evolution | 标量损失 |
| `sac` | 连续动作强化学习 | 观测和奖励 |
| `ppo` | 连续动作强化学习 | 观测和奖励 |

`geometric` 和 `response_alignment` 是解析基线，不进行迭代。其余方法从同一几何相位初始化。主对比采用统一的 `60 s` 优化墙钟窗口与同一 GPU 响应基；`15000` 场评估仅是保护上限，不是强制用满的预算。已收敛或停滞的方法允许提前停止，避免无效迭代扭曲效率。`adjoint` 在主表中固定为本场景效果最佳的 Adam 更新器；L-BFGS-B 仅保留在梯度更新器消融实验中。

## 本次结果

| 算法 | Best loss | 对比度 | 场评估次数 | 停止时耗时 |
|---|---:|---:|---:|---:|
| geometric | 1788.425 | 0.306 | 1 | 30.12 s* |
| response_alignment | -1220.675 | 1.883 | 1 | 0.06 s |
| adjoint (Adam) | **-1872.539** | **2.814** | 1701 | **12.54 s** |
| gabs | -1592.518 | 2.468 | 11521 | 44.80 s |
| spsa | -158.063 | 1.364 | 14322 | 60.01 s |
| cmaes | -1779.361 | 2.674 | 12110 | 60.01 s |
| lshade | -1660.869 | 2.528 | 14142 | 60.00 s |
| sac | 1149.294 | 0.462 | 725 | 60.19 s |
| ppo | 981.026 | 0.528 | 8482 | 50.80 s |

`*` geometric 的时间包含该进程首次 GPU kernel JIT 和响应基设备端初始化，不属于优化工作；因此不用于算法墙钟比较。

“停止时耗时”只描述严格收敛、预算耗尽或训练结束的时刻，不作为不同终点质量的主效率排名。特别是 Adam 在到达高质量解后仍继续少量精修，以验证稳定停滞；L-SHADE 与 CMA-ES 在 60 s 结束时仍未收敛。

RL 从零开始训练的总成本被完整计入，不应与预训练后单次推理的成本混淆。SAC 在训练期耗尽 60 s 后未进行额外策略回放，表中记录训练期间最佳已评估候选，并标记为 `training_time_budget`。

### 墙钟时间解读

本表已使用 GPU 常驻响应基、GPU loss/cotangent/VJP 的当前实现重新运行。CuPy JIT 使用内存缓存以绕过用户级 kernel cache 的文件系统阻塞；首次 kernel 编译不计入后续算法的稳态优化时间。

最终 loss 不是唯一效率指标。下表记录各方法首次达到同一质量门槛的时间与场评估次数；未达到表示在 60 s 窗口内未达到该质量。

| 方法 | 达到 `-1220.675` | 达到 `-1593.498` | 达到 `-1800` | 达到 `-1860` |
|---|---|---|---|---|
| adjoint (Adam) | 1.17 s / 53 次 | 1.46 s / 89 次 | **2.00 s / 163 次** | **3.44 s / 350 次** |
| CMA-ES | 17.51 s / 2734 次 | 29.30 s / 5762 次 | 未达到 | 未达到 |
| L-SHADE | 18.40 s / 3900 次 | 41.44 s / 9453 次 | 未达到 | 未达到 |
| GABS | 9.58 s / 2832 次 | 未达到 | 未达到 | 未达到 |
| SPSA / SAC / PPO | 未达到 | 未达到 | 未达到 | 未达到 |

共同质量门槛在实验前由已有解析基线的 `-1220.675`、`-1593.498`、任务高质量线 `-1800` 与近 Adam 终局质量线 `-1860` 定义；它们不是从某一算法停止时反向挑选。主效率结论应读取这一表：Adam 达到 `-1860` 只需 `3.44 s`，而其 `12.54 s` 是严格停滞确认的辅助信息。场评估次数保留为真实装置闭环时的交互成本指标，但不会被强制拉齐。该表仍是单随机种子结果，后续应报告多随机种子的均值、标准差和 P95。

## 目录

```text
algorithm_comparison/
  config/
    phase_oblique_reflecting_x_12x12.yaml
  results/
    phase_oblique_reflecting_x_12x12/
      summary.csv / summary.json
      quality_summary.csv / quality_summary.json
      loss_dashboard.png
      <algorithm>/
        config.yaml
        result.npz
        loss_curve.png
        policy.zip / policy.json       # 仅 SAC、PPO
  run_comparison.py
```

`result.npz` 包含最终相位、最佳已评估相位、复声压场、振幅场、目标场、几何基线场、loss 历史和运行元数据。`summary.csv` 是便于表格软件读取的汇总，`summary.json` 保留相同的结构化信息。

## 复现

从仓库根目录运行：

```powershell
python algorithm_comparison/run_comparison.py
```

该命令将重新运行所有算法，并覆盖 `algorithm_comparison/results/phase_oblique_reflecting_x_12x12/`。上传包不包含大型相位响应基缓存，首次运行会自动构建，后续运行自动复用本地缓存。需要安装项目基础依赖；SAC、PPO 与 CMA-ES 还需要 `src/baselines/requirements.txt` 中的可选依赖。

总图 `loss_dashboard.png` 将所有算法叠加到同一坐标轴：每种颜色的透明细线是原始 current loss，实线是自适应滚动中位数，虚线是截至该评估次数的 best-so-far。滚动中位数只平滑局部噪声，不会像全程均值那样把早期搜索阶段混入后期收敛值。

从既有 `result.npz` 重建共同质量统计，无需重跑优化：

```powershell
python src/compare.py --config algorithm_comparison/config/phase_oblique_reflecting_x_12x12.yaml --output-dir algorithm_comparison/results --rebuild-quality-summary
```
