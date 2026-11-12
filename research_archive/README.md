# Research Archive

This directory keeps historical experiment evidence that is not part of the current hand-phase surrogate training loop, but is useful for papers, reports, and presentations.

## Contents

```text
algorithm_comparison/
gradient_ablation/
obstacle_displacement_study/
assets/
```

- `algorithm_comparison/`: baseline comparison across phase optimization algorithms such as geometric, response alignment, adjoint, CMA-ES, L-SHADE, SPSA, SAC, PPO, and GABS.
- `gradient_ablation/`: optimizer and analytic-gradient ablation evidence.
- `obstacle_displacement_study/`: early differentiability and obstacle-displacement evidence for plate, sphere, and hand geometries.
- `assets/`: legacy figures/photos used by earlier reports and presentations.

These artifacts are intentionally archived rather than deleted because they support later paper writing:

- motivation for replacing strict BEM/PDE solves with a neural surrogate;
- baseline and ablation tables;
- convergence plots and qualitative field figures;
- historical evidence for differentiability and obstacle sensitivity.

Do not place current training checkpoints or generated datasets here. Current active outputs belong under `outputs/`.
