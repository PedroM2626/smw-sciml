[Contents](../../README.md#contents) · [2 →](02-introduction-and-scientific-motivation.md)

## 1. Project Overview & Abstract

This research provides an empirical investigation into the impact of embedding known discrete kinematic constraints and structural physical priors (*Physics-Informed Machine Learning* — PIML / PINN) into predictive world modeling for discrete-time dynamic systems. Using *Super Mario World* (SNES, 1990) executed within a high-throughput headless emulation environment with direct Random Access Memory (RAM) telemetry (free of computer vision or pixel rendering pipelines), we benchmark four distinct neural network paradigms:
1. **Statistical Multilayer Perceptron (MLP)**: Pure supervised black-box baseline;
2. **Statistical Recurrent Neural Network (LSTM)**: Sequential model with latent temporal memory;
3. **Soft-Constrained PINN**: Dense network penalized via Lagrangian regularization of kinematic and boundary residuals in the objective loss;
4. **Hard-Constrained Residual PINN**: Hybrid inductive architecture where analytical kinematic integration and saturation limits are embedded directly into the PyTorch computational graph, delegating only the estimation of unmodeled contact and force residuals to the neural layers.

All evaluations were conducted on strictly genuine transitions recorded directly from console Working RAM (WRAM) during interactive gameplay (Game Mode `$14` — Yoshi's Island 1), without synthetic or fabricated data.

---
