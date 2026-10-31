[← 8](08-empirical-results-and-comparative-benchmark-tables.md) · [Contents](../../README.md#contents) · [10.1 →](10-01-quantitative-answer-how-much-does-knowing-the.md)

## 9. Visual Analysis of Trajectories and Convergence

All figures were generated directly from empirical runs and reside in `results/figures/`:

### 9.1 Training Convergence Dynamics
The Hard Residual PINN initializes training near a loss of 1.0, achieving optimal convergence within 5 epochs, whereas statistical models require dozens of epochs to reconcile coordinate scales.

![Training Convergence](../../results/figures/training_convergence.png)

---

### 9.2 Long-Horizon Rollout Dispersion (Drift Comparison)
Euclidean spatial drift across 120 frames of open-loop autoregressive simulation:

![Rollout Drift Comparison](../../results/figures/rollout_drift_comparison.png)

---

### 9.3 2D Spatial Trajectory Traversal $(X, Y)$
Predicted character paths versus ground-truth SNES console trajectory during interactive gameplay:

![Trajectory in 2D Space](../../results/figures/trajectory_2d_space.png)

---

### 9.4 Sample Efficiency Pareto Frontiers
The Hard Residual PINN maintains near-optimal performance even when data drops to $N = 200$, whereas the MLP degrades exponentially when $N \lt 2,500$.

| Test Error vs. Dataset Size (MSE) | Trajectory Drift vs. Dataset Size (Drift) |
| :---: | :---: |
| ![Sample Efficiency MSE](../../results/figures/sample_efficiency_mse.png) | ![Sample Efficiency Drift](../../results/figures/sample_efficiency_drift.png) |

---
