[← 10.25](10-25-multi-iteration-interactive-dagger-policy-831-px-progress.md) · [Contents](../../README.md#contents) · [10.27 →](10-27-master-algorithm-comparison-table-world-models-mpc.md)

### 10.26 Comprehensive Ablation Study (Clamping, Horizon Drift & CEM Sensitivity)

To isolate the individual contribution of each component of the Hard PINN framework, we executed three systematic ablation studies (`src/evaluation/ablation_benchmark.py`):

1. **Velocity Saturation Clamping Ablation:**
   - Evaluated Hard Residual PINN with vs. without velocity saturation clamping $[-v_{\max}, v_{\max}]$.
   - Clamped MSE: **22.61** vs. Unclamped MSE: **16.58** on single step, but unbounded models exhibit risk of runaway kinematic extrapolation under out-of-distribution control sequences.
2. **Multi-Step Autoregressive Horizon Degradation Curve ($H \in \{1, 5, 15, 30, 60, 120\}$):**
   - At $H=1$: Hard PINN achieves **6.08** MSE vs **64,277** for MLP, **138,449** for LSTM, and **56,227** for Soft PINN (an error reduction of **10,571×**).
   - At $H=30$: Hard PINN maintains **845.09** MSE vs **70,622** for MLP and **140,936** for LSTM (an error reduction of **83×**).
   - **Kinematic Violation Rate:** Statistical baselines violated physics on **99.4% to 100.0%** of frames, while Hard PINN maintained **3.6%** (analytical zero on position updates).
3. **CEM MPC Planning Parameter Sensitivity ($N \in \{32, 64, 128, 256, 512\}$):**
   - Candidate counts from $N=32$ to $N=512$ scaled with GPU parallelism, maintaining **~40.5 ms / step** (24.7 FPS) while increasing trajectory reward from 88.1 to 88.9.
- **Visual Artifact:** Publication-ready 3-panel figure saved to `results/figures/ablation_study_comparison.png` and raw metrics in `results/ablation_benchmark_metrics.json`.

![Ablation Study Comparison](../../results/figures/ablation_study_comparison.png)

---
