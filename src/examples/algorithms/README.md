# 相位算法配置示例

每次运行选择一个顶层 `algorithm`：

```powershell
python src/main.py --config src/examples/algorithms/spsa.yaml
```

所有算法共用同一个固定场景、目标场和 `training.loss_type`。算法专属参数只放在 `algorithm_options`。

| 算法 | 配置名 | 专属参数 |
|---|---|---|
| 伴随梯度 | `adjoint` | `optimizer`、`learning_rate`、`gradient_check`、`gradient_check_step` |
| 几何相位 | `geometric` | 无 |
| 响应对齐 | `response_alignment` | 无 |
| 贪心坐标搜索 | `gabs` | `phase_levels` |
| 随机近似 | `spsa` | `learning_rate`、`perturbation`、`alpha`、`gamma` |
| CMA-ES | `cmaes` | `sigma`、`population_size` |
| L-SHADE / Differential Evolution | `lshade` | `population_size`、`min_population_size`、`memory_size`、`p_best_rate`、`convergence_patience`、`convergence_relative_tolerance` |
| Soft Actor-Critic | `sac` | `episode_steps`、`evaluation_steps`、`action_scale`、`reward_scale`、`total_timesteps`、`checkpoint` 等 |
| Proximal Policy Optimization | `ppo` | 与 `sac` 相同，另加 `n_steps` |

`gabs.yaml`、`spsa.yaml`、`lshade.yaml` 和 `sac.yaml` 提供了可复制的完整配置。将其中的 `algorithm` 和 `algorithm_options` 替换为表中的对应值，即可切换到其余算法。

SAC、PPO、CMA-ES 需要可选依赖；L-SHADE 仅依赖 NumPy：

```powershell
python -m pip install -r src/baselines/requirements.txt
```

SAC/PPO 的 `run_mode: "train"` 会训练并保存策略；改为 `run_mode: "evaluate"` 时，会加载同一 `checkpoint`，并校验场景任务指纹和环境参数。策略训练时间与测试时相位调整时间会分别写入结果元数据。
