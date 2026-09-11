# 五分钟项目答辩

正式入口：[index.html](index.html)。

可直接用 Edge 或 Chrome 打开，不需要启动服务器，不依赖网络。保持 `presentation.css`、`presentation.js`、`media.js` 和 `assets/` 与 HTML 同目录。

备用投影文件：[defense_5min.pdf](defense_5min.pdf)。PDF 为八页固定 16:9 画幅，不含控制栏和演讲备注。

## 内容

| 页 | 主题 | 建议时长 |
| --- | --- | --- |
| 1 | 问题与主张 | 20 秒 |
| 2 | 三维 Helmholtz 求解器与边界凝聚 | 45 秒 |
| 3 | 响应基、目标函数与伴随 VJP | 60 秒 |
| 4 | 我们的方法与忽略障碍物 baseline 的效果对比（待补图） | 30 秒 |
| 5 | 原始算法 Loss 对比图（九种方法） | 30 秒 |
| 6 | 原始优化器消融 Loss 曲线（五种更新器） | 25 秒 |
| 7 | 共同质量门槛的数值比较 | 40 秒 |
| 8 | 16 × 16 硬件部署及结论 | 25 秒 |

合计 4 分 35 秒，预留 25 秒缓冲。每页均附中文演讲备注。

## 演示操作

- 左右方向键、PageUp / PageDown：翻页；空格：下一页。
- `1` 至 `8`：直接跳转；Home / End：首页 / 末页。
- `F`：全屏或退出全屏。
- `N`：查看当前页讲稿；`O`：章节目录；Esc：关闭弹窗。
- 移动鼠标显示底部图标工具栏。打印图标可重新导出 PDF。
- 触屏支持左右滑动。窄屏按固定比例缩放，不重新排列幻灯片。

## 数据与表述边界

1. 第一页等幅线从归档 `adjoint/result.npz` 提取。不是 COMSOL 图片或实测数据。
2. 封面切面为 `y = 90 mm`，显示 `z ≥ 12 mm`。原型为 12 × 12 阵列、41³ 网格、+x 反射，无障碍物。不得当成障碍物实验证据。第四页已留空，等待真实效果对比图。
3. 第七页左图读取算法比较的 `quality_summary.csv`，统一门槛为 `−1593.498`；右图读取独立消融的 `quality_summary.csv`，统一门槛为 `−1860`。秒数不能在两组独立实验之间直接混用。
4. 图中加速比由未四舍五入的原始秒数计算。数据为归档单次、单场景、单种子结果；没有多次重复统计，因此不画虚构误差棒、不声称普遍最优。
5. 场评估次数指响应基场合成及目标计算，不等同于每次重新求解一次大型 PDE。时间取自优化日志，不含响应基构建，也不代表陌生场景端到端时延。
6. 项目已开展 COMSOL 验证，但当前未提供原始对照图和定量统计。因此第四页列出核验口径，不给出虚构误差。最终答辩前应补充相同坐标、单位和激励下的实际对照。
7. 实物阵列已具备；标定与声场闭环尚未完成。第八页只画明确标注的阵列拓扑示意，不使用生成照片冒充装置。
8. 相位梯度采用代码约定 `dL = Re(gᴴ du)`，正确表达为 `Im(conj(z) ⊙ (Gᴴg))`。旧大纲的负号已同步修正。
9. 第五、六页直接使用归档 `loss_dashboard.png` 的完整副本，没有裁切图例、坐标或重画曲线。横轴为场评估次数；实线是当前损失的滑动中位数，虚线是历史最优值，浅线是原始损失。它们不是多随机种子均值或置信区间。

## 放入效果图

第四页左侧为现有 baseline，右侧为我们的方法。默认 `media.js` 中两项为空，不请求不存在的图片，也不生成替代效果图。

将自己的两幅图片放到 `ppt/assets/` 的合适子目录，然后在 [media.js](media.js) 中填写相对路径，例如：

```javascript
window.DEFENSE_MEDIA = window.DEFENSE_MEDIA || {
  baseline: "assets/effects/baseline.png",
  ours: "assets/effects/ours.png",
};
```

刷新 HTML 后会自动填入左右预留区，以 `object-fit: contain` 完整显示，不裁剪或拉伸。留空仍显示“效果图待补”；错误路径会显示加载失败。

对比必须让两组相位在同一个含障碍物场景中复算或实测。所谓“无障碍物方法”指 baseline 在求相位时不建模障碍物，不能直接把无障碍物的评估结果与含障碍物的结果混比。具体 baseline 名称、边界条件、激励幅值、单位、坐标、色标以及测量/仿真来源，应随图补充。

补图后 PDF 不会自动变化，需要重新运行 `python ppt/verify_presentation.py` 或在浏览器中打印。

## 原始来源

- [算法达标统计](../algorithm_comparison/results/phase_oblique_reflecting_x_12x12/quality_summary.csv)
- [原始算法 Loss 图](../algorithm_comparison/results/phase_oblique_reflecting_x_12x12/loss_dashboard.png)
- [原始优化器消融 Loss 图](../gradient_ablation/results/phase_oblique_reflecting_x_12x12/loss_dashboard.png)
- [消融达标统计](../gradient_ablation/results/phase_oblique_reflecting_x_12x12/quality_summary.csv)
- [归档声场](../algorithm_comparison/results/phase_oblique_reflecting_x_12x12/adjoint/result.npz)
- [原型配置](../algorithm_comparison/results/phase_oblique_reflecting_x_12x12/adjoint/config.yaml)
- [相位 VJP](../src/algorithm/training/phase_response_basis.py)
- [损失函数](../src/algorithm/training/phase_adjoint_optimizer.py)
- [边界凝聚](../src/algorithm/solvers/boundary_condensed_solver.py)
- [硬件状态与标定手册](../src/hardware/README.md)

## 重建与验证

从仓库根目录执行：

```powershell
python ppt/build_evidence.py
python ppt/verify_presentation.py
```

`build_evidence.py` 依赖 NumPy 和 Matplotlib，仅提取数值与等值线，不重新运行求解器；输出离线数据文件 `assets/evidence.js`。

`verify_presentation.py` 使用 Playwright、Pillow 和本机 Edge，检查五种视窗下全部八页及效果图加载，保存截图、检查报告与 PDF。依赖用于维护，不影响直接打开 HTML。

`assets/lucide.min.js` 是本地保存的 Lucide 0.468.0 图标库，许可证保存在 `assets/lucide.LICENSE`；没有在线运行时依赖。

旧入口 `defense_5min.html` 已改为自动跳转至新版 `index.html`，避免误用初稿。
