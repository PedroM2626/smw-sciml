[← 7](07-experimental-protocol-data-partitioning-hyperparameters.md) · [Contents](../../README.md#contents) · [9 →](09-visual-analysis-of-trajectories-and-convergence.md)

## 8. Empirical Results and Comparative Benchmark Tables

All numerical values below are extracted directly from empirical benchmark logs in `results/benchmark_metrics.json` and `results/sample_efficiency_metrics.json`.

### 8.1 Single-Step Accuracy on Independent Test Set ($N_{\text{test}} = 1,356$)

| Evaluated Architecture | Paradigm | Test Loss (Data MSE) | Kinematic Residual ($\|\Delta X - \frac{v_x}{16}\|^2$) | Training Time (s) | Stopping Epoch |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Statistical MLP** | Supervised Black-Box | 16.4717 | 17,561.24 | 4.16s | 35 (final epoch) |
| **Statistical LSTM** | Recurrent Sequence | 39.2194 | 37,361.16 | 1.73s | 11 (early stop) |
| **Soft-Constrained PINN** | Loss Penalty Regularization | 53.8167 | 108,520.99 | 6.60s | 35 (final epoch) |
| **Hard Residual PINN** | **Hard Inductive Bias** | **0.5783** | **0.0019** | 4.66s | 35 (final epoch) |

---

### 8.2 Long-Horizon Stability: 120-Frame Open-Loop Autoregressive Rollout (2 Seconds)

| Architecture | Mean Trajectory Drift (px) | Final Drift at Frame 120 (px) | Kinematic Violations (Frames) | Velocity Bound Violations |
| :--- | :---: | :---: | :---: | :---: |
| **Statistical MLP** | 152.72 px | 69.75 px | 118 / 120 (**98.3%**) | 40 / 120 (33.3%) |
| **Statistical LSTM** | 169.99 px | 126.16 px | 120 / 120 (**100.0%**) | 0 / 120 (0.0%) |
| **Soft-Constrained PINN** | 220.05 px | 231.18 px | 120 / 120 (**100.0%**) | 0 / 120 (0.0%) |
| **Hard Residual PINN** | **50.87 px** | 78.13 px | **0 / 120 (0.0%)** | 0 / 120 (0.0%) |

---

### 8.3 Systematic Sample Efficiency Study (Data Pareto Curve)

| Training Sample Size ($N$) | Equivalent Playtime | Statistical MLP (Test MSE) | Soft-PINN (Test MSE) | Hard Residual PINN (Test MSE) | Hard PINN Advantage Over MLP |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 200$** | ~3.3 seconds | 76.4664 | 76.7718 | **0.6764** | **113.0x lower error** |
| **$N = 500$** | ~8.3 seconds | 73.5304 | 73.7688 | **0.5987** | **122.8x lower error** |
| **$N = 1,000$** | ~16.6 seconds | 66.6258 | 66.8568 | **0.5983** | **111.4x lower error** |
| **$N = 2,500$** | ~41.6 seconds | 45.4729 | 52.0445 | **0.5868** | **77.5x lower error** |
| **$N = 5,000$** | ~83.3 seconds | 12.4001 | 52.6735 | **0.5779** | **21.5x lower error** |

#### Autoregressive Rollout Drift vs. Training Sample Size:

| Training Sample Size ($N$) | Statistical MLP (Rollout Drift) | Soft-PINN (Rollout Drift) | Hard Residual PINN (Rollout Drift) | Hard PINN Drift Reduction |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 200$** | 330.38 px | 329.63 px | **41.89 px** | **7.9x lower drift** |
| **$N = 500$** | 319.54 px | 315.97 px | **12.41 px** | **25.7x lower drift** |
| **$N = 1,000$** | 293.04 px | 286.24 px | **15.73 px** | **18.6x lower drift** |
| **$N = 2,500$** | 223.09 px | 230.00 px | **35.91 px** | **6.2x lower drift** |
| **$N = 5,000$** | 85.90 px | 232.39 px | **16.25 px** | **5.3x lower drift** |

---

### 8.4 Multi-Seed Statistical Significance Benchmark ($K = 10$ Seeds)

To guarantee academic rigor and verify that the results are not artifacts of seed variance, we evaluated the architectures across $K = 10$ independent random partitions ($S \in \{42, \dots, 51\}$). All metrics report sample mean ± sample standard deviation ($\mu \pm \sigma$):

| Architecture | Test Loss (Data MSE) | Kinematic Residual ($\|\Delta X - \frac{v_x}{16}\|^2$) | 120-Frame Mean Drift (px) | Kinematic Violations (Frames) |
| :--- | :---: | :---: | :---: | :---: |
| **Statistical MLP** | $55.06 \pm 16.40$ | $153,243.44 \pm 87,149.73$ | $172.06 \pm 12.30\text{ px}$ | 120 / 120 (**100.0%**) |
| **Soft-Constrained PINN** | $49.97 \pm 10.02$ | $126,676.65 \pm 60,421.21$ | $200.66 \pm 45.80\text{ px}$ | 120 / 120 (**100.0%**) |
| **Hard Residual PINN** | **$0.65 \pm 0.15$** | **$0.0014 \pm 0.0003$** | **$86.44 \pm 37.65\text{ px}$** | **0 / 120 (0.0%)** |

#### Formal Statistical Hypothesis Testing (Paired Tests across 10 Seeds):
We conducted formal hypothesis testing comparing the **Hard Residual PINN** against the baselines:
1. **Hard PINN vs. Statistical MLP:**
   - **Test MSE:** Student's paired $t$-test yields $t = -10.50$, **$p = 2.39 \times 10^{-6}$ ($p \lt 0.001$)**; Wilcoxon signed-rank test yields $W = 0.0$, **$p = 0.002$ ($p \lt 0.01$)**; Cohen's $d_z = -3.32$ (very large effect).
   - **Rollout Mean Drift:** Student's paired $t$-test yields $t = -6.64$, **$p = 9.46 \times 10^{-5}$ ($p \lt 0.001$)**; Wilcoxon $p = 0.002$; Cohen's $d_z = -2.10$.
   - **Multi-Start Drift:** $t = -5.57$, **$p = 3.49 \times 10^{-4}$**; Wilcoxon $p = 0.002$; Cohen's $d_z = -1.76$.
2. **Hard PINN vs. Soft-Constrained PINN:**
   - **Test MSE:** Student's paired $t$-test yields $t = -15.58$, **$p = 8.09 \times 10^{-8}$ ($p \lt 0.001$)**; Wilcoxon signed-rank test yields $W = 0.0$, **$p = 0.002$ ($p \lt 0.01$)**; Cohen's $d_z = -4.93$ (very large effect).
   - **Rollout Mean Drift:** Student's paired $t$-test yields $t = -7.31$, **$p = 4.51 \times 10^{-5}$ ($p \lt 0.001$)**; Wilcoxon $p = 0.002$; Cohen's $d_z = -2.31$.
   - **Multi-Start Drift:** $t = -6.50$, **$p = 1.11 \times 10^{-4}$**; Wilcoxon $p = 0.002$; Cohen's $d_z = -2.06$.

The empirical results confirm with high statistical significance that the Hard Residual PINN decisively outperforms both the unconstrained black-box MLP and the Lagrangian soft penalty PINN.

---
