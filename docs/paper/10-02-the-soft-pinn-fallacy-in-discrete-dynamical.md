[← 10.1](10-01-quantitative-answer-how-much-does-knowing-the.md) · [Contents](../../README.md#contents) · [10.3 →](10-03-analysis-of-lstm-performance-and-the-markovian.md)

### 10.2 The Soft PINN Fallacy in Discrete Dynamical Systems
One of the most consequential findings of this study is elucidating why soft loss-regularized PINNs (*Soft PINNs*, Raissi et al., 2019) **fail in discrete game dynamics**:
* **Gradient Stiffness:** In continuous PDEs, differential operators yield smooth loss gradients. In 60 Hz discrete systems with fixed-point arithmetic and contact discontinuties, minor fraction errors cause kinematic penalties to explode (a kinematic term reaching $\sim 10^{5}$), dwarfing data loss
(order $10^{1}$).
* **Pareto Gradient Conflict:** AdamW expends almost its entire gradient budget satisfying $\Delta X - v_x/16.0 = 0$, starving parameters responsible for learning force residuals and contact logic.
* **Empirical Outcome:** Soft PINN performed **worse than the unconstrained MLP** (53.82 vs. 16.47 MSE) and still incurred 100% rollout violations. In discrete dynamics, hard inductive constraints are indispensable.

---
