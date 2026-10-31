[← 10.16](10-16-out-of-distribution-ood-zero-shot-cross-stage-generalization-stage-a.md) · [Contents](../../README.md#contents) · [10.18 →](10-18-genuine-multi-entity-dataset-supervised-hazard-dynamics-training.md)

### 10.17 Synchronized Multi-Model Visualization (Real SNES vs. PINN vs. MLP)

To provide clear qualitative insight into open-loop degradation, we generated synchronized 120-frame (2.0-second) autoregressive rollouts starting from identical initial conditions in WRAM (`src/evaluation/render_comparison_animation.py`):

1. **Real SNES Ground Truth (Black Curve):** Authentic player movement with smooth jumping arcs and ground friction.
2. **Hard Residual PINN (Green Dashed Curve):** Perfectly tracks the physical manifold with minimal drift and zero kinematic violations (0.0%).
3. **Statistical MLP (Red Dotted Curve):** Drifts severely as errors compound, floating unnaturally above ground and violating the laws of motion.

![Model Comparison Trajectory Composite](../../results/figures/model_comparison_trajectory_composite.png)

*An animated GIF showing synchronized frame-by-frame trajectory progression is maintained at [`results/figures/model_comparison_animation.gif`](../../results/figures/model_comparison_animation.gif).*

---
