[← 10.10](10-10-deep-ensemble-of-hard-pinns-epistemic-uncertainty.md) · [Contents](../../README.md#contents) · [10.12 →](10-12-closed-loop-active-model-based-policy-optimization-online-mbpo.md)

### 10.11 Spatial Translation-Invariant PINN Dynamics

Newtonian mechanics and 65816 CPU assembly routines for velocity integration, air drag, and jumping are strictly invariant under spatial translation:

$$F = m \cdot a \quad (\text{Independent of global horizontal coordinate } X)$$

In standard architectures, passing absolute $X_t \in [0, 2500]$ into dense layers forces the network to overfit to the terrain profile of *Yoshi's Island 1*. To establish true cross-level generalization, we implemented **Translation-Invariant PINN Dynamics** (`src/models/pinn_invariant.py`):
- The neural force head receives only local kinematics and actions: $[v_x, v_y, c_{\text{ground}}, c_{\text{ceiling}}, c_{\text{left}}, c_{\text{right}}, a_t]$.
- Absolute coordinates are integrated strictly through the analytical kinematic accumulator $\hat X_{t+1} = X_t + \hat v_{x, t+1} / 16.0$.

#### Spatial Equivariance Theorem:

$$\forall C \in \mathbb{R}, \quad f_\theta(s + [C, 0, \dots], a) = f_\theta(s, a) + [C, 0, \dots]$$
Verified via automated unit tests (`tests/test_pinn_invariant.py`) with numerical error $|\Delta - C| \lt 10^{-5}\text{ px}$, guaranteeing zero-shot transfer across any stage regardless of coordinate origin.

---
