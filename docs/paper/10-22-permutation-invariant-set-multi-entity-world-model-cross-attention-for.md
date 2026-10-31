[← 10.21](10-21-empirical-grounding-of-tilemap-pinn-wram-7e-c800.md) · [Contents](../../README.md#contents) · [10.23 →](10-23-amortized-policy-distillation-from-live-mpc-decisions.md)

### 10.22 Permutation-Invariant Set Multi-Entity World Model (Cross-Attention for N Sprites)

To remove the single-hazard constraint ($K=1$), we implemented `SetMultiEntityPINNDynamics` (`src/models/pinn_set_multi_entity.py`):
- **Cross-Attention & Deep Sets:** Mario's 8D kinematic state acts as Query $Q$, while a variable set of active sprites $\{e_1, \dots, e_K\}$ act as Keys/Values with padding masks for inactive slots.
- **Permutation Invariance & Equivariance:** Permuting the order of sprite slots produces identical Mario next states and permutes entity predictions accordingly.
- **Exact Multi-Entity Kinematic Law:** Enforces analytical relative kinematic integration simultaneously across all $K$ entities:

  $$\Delta\hat X_{i, t+1} = \Delta X_{i, t} + \frac{\hat v_{xi, t+1} - \hat v_{xm, t+1}}{16.0}$$
  guaranteeing 0.0% kinematic violation for all entities regardless of entity count.
- **Unit Test Suite:** Fully verified by `tests/test_set_multi_entity.py` (100% passing tests).

---
