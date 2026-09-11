# 五分钟项目答辩

正式入口：[index.html](index.html)。

可直接用 Edge 或 Chrome 打开，不需要启动服务器，不依赖网络。保持 `presentation.css`、`presentation.js` 和 `assets/` 与 HTML 同目录。

备用投影文件：[defense_5min.pdf](defense_5min.pdf)。PDF 为十七页固定 16:9 画幅，不含控制栏和演讲备注。

## 内容

| 页 | 主题 | 建议时长 |
| --- | --- | --- |
| 1 | 项目首页与核心主张 | 15 秒 |
| 2 | 章节目录 | 10 秒 |
| 3 | XR、遥操作、医疗与空间引导 | 20 秒 |
| 4 | 可编程超声场的技术基础 | 15 秒 |
| 5 | 复杂环境声场控制的相关工作 | 15 秒 |
| 6 | 从场景到硬件的技术路线 | 10 秒 |
| 7 | 三维 Helmholtz 求解器与边界凝聚 | 25 秒 |
| 8 | COMSOL 独立交叉验证 | 20 秒 |
| 9 | \(B\)、\(G\) 与响应基的分批构建 | 25 秒 |
| 10 | 复声场梯度 \(c\) 的定义与来源 | 15 秒 |
| 11 | VJP 与全部阵元相位梯度 | 20 秒 |
| 12 | 几何相位与物理优化的声场对比 | 15 秒 |
| 13 | 原始算法 Loss 对比图 | 15 秒 |
| 14 | 更新器消融 Loss 曲线 | 15 秒 |
| 15 | 共同质量门槛的数值比较 | 20 秒 |
| 16 | SonicSurface 实物与相位标定 | 30 秒 |
| 17 | 结论与下一步 | 15 秒 |

合计 5 分钟。每页均附中文演讲备注。

## 演示操作

- 左右方向键、PageUp / PageDown：翻页；空格：下一页。
- `1` 至 `9`：直接跳转到对应页；第 10–17 页可用方向键或目录进入；Home / End：首页 / 末页。
- `F`：全屏或退出全屏。
- `N`：查看当前页讲稿；`O`：章节目录；Esc：关闭弹窗。
- 移动鼠标显示底部图标工具栏。打印图标可重新导出 PDF。
- 触屏支持左右滑动。窄屏按固定比例缩放，不重新排列幻灯片。

## 素材与数据

1. 首页和第十六页使用项目实拍图 `实物照片.JPG` 的演示版裁切。
2. 第十六页使用 `相位对齐调试.png` 展示 AcousticField 中的相位标定过程。
3. 第四、五页使用 `ppt_ref/` 中整理的论文原图，并在页脚标明作者、年份和 DOI。
4. 第八页读取 `comsol_comparison` 的三维场图和跨方法目标对比度统计图。
5. 第十二页从归档 `adjoint/result.npz` 生成几何相位与物理优化的同切面、同色标对比。
6. 第十三、十四页使用归档 Loss 曲线；第十五页读取两组 `quality_summary.csv`。
7. 优化数值实验为 12 × 12、41³ 网格、+x 反射场景；当前硬件平台为 16 × 16 SonicSurface。

## 原始来源

- [非接触超声触觉](https://doi.org/10.1109/TOH.2010.4)
- [GS-PAT 多点声场](https://doi.org/10.1145/3386569.3392492)
- [Diff-PAT 可微声学全息](https://doi.org/10.1038/s41598-021-91880-2)
- [SonicSurface](https://doi.org/10.3390/app11072981)
- [中空超声触觉综述](https://doi.org/10.1109/TOH.2020.3018754)
- [任意散射体声学全息](https://doi.org/10.1126/sciadv.abn7614)
- [障碍物下声压场重建](https://doi.org/10.1109/TOH.2023.3309975)
- [手部散射与触觉感知](https://doi.org/10.1145/3706599.3720287)
- [COMSOL 交叉验证说明](../comsol_comparison/README.md)
- [COMSOL 正演指标](../comsol_comparison/extended/results/extended_summary.csv)
- [COMSOL 相位研究指标](../comsol_comparison/phase_study/results/phase_summary.csv)
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

`verify_presentation.py` 使用 Playwright、Pillow 和本机 Edge，检查五种视窗下全部十七页及图片加载，保存截图、检查报告与 PDF。依赖用于维护，不影响直接打开 HTML。

`assets/lucide.min.js` 是本地保存的 Lucide 0.468.0 图标库，许可证保存在 `assets/lucide.LICENSE`；没有在线运行时依赖。

旧入口 `defense_5min.html` 已改为自动跳转至新版 `index.html`，避免误用初稿。
