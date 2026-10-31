[← 10.41](10-41-neural-operator-baselines-deeponet-on-discrete-engine-dynamics.md) · [Contents](../../README.md#contents) · [10.43 →](10-43-symbolic-regression-as-an-inverse-problem-method-discovering.md)

### 10.42 Physics-Constrained DeepONet and the Fourier Neural Operator

Section 10.41 closed with two open branches: (a) a **physics-constrained neural operator** - a DeepONet whose branch/trunk factorization is confined to force and contact residuals while the Section 4 kinematics integrate them analytically - and (b) the **spectral branch of operator learning**, the Fourier Neural Operator (FNO; Li et al., 2021, ICLR), over the same sensor readings. This section closes both branches with genuine unified-protocol runs recorded in `results/operator_benchmark_metrics.json`.

**Formulation A - Physics-Constrained DeepONet (`PhysicsConstrainedDeepONetDynamics`, 52,742 parameters).** The operator no longer predicts the next state directly; it predicts the same *residual space* the Hard Residual PINN learns - velocity increments and contact flags - through a basis expansion over the residual-output query grid:

$$[\delta v_x,\; \delta v_y,\; \hat c_{\text{aux}}](../../q) = \langle b([s_t, a_t]),\; \tau(y_q) \rangle + b_0[q],\qquad q \in \{0, \dots, 5\}$$

and the engine's discrete integration is applied analytically in the computation graph, exactly as
in Section 5.4, one rule per channel:

* $\hat v_x = \mathrm{clamp}(v_x + \delta v_x, \pm 72)$
* $\hat v_y = \mathrm{clamp}(v_y + \delta v_y, -80, 64)$
* $\hat X_{t+1} = X_t + \hat v_x / 16.0$
* $\hat Y_{t+1} = Y_t + \hat v_y / 16.0$

The operator learns forces; the engine rule integrates them. The kinematic consistency residual is identically zero **by construction**, so this hybrid cannot violate the discrete update identity regardless of what the branch/trunk nets learn.

**Formulation B - FNO (`FNODynamics`, 14,537 parameters: width 32, 6 Fourier modes, 2 spectral layers).** The state-action reading $u = [s_t, a_t]$ is collocated on the uniform 14-sensor lattice $x_i = i/13 \in [0,1]$; each pair $[u(x_i), x_i]$ is lifted to a 32-channel latent field, two spectral convolution blocks act on it (FFT $\to$ learned complex weights on the 6 lowest modes $\to$ inverse FFT $+$ pointwise bypass, GELU), and the resulting scalar field $f$ on $[0,1]$ is decoded at the canonical output-channel queries $y_q$ by linear interpolation - the FNO's continuous, grid-independent evaluation mode applied to discrete channel coordinates:

$$\hat s_{t+1}[q] = f\!\left(y_q\right) + b_0[q],\qquad f = \mathcal{F}^{-1}\Big[\sum_{|k| \le k_{\max}} R_k \cdot \mathcal{F}\{V_L\}[k]\Big]$$

```
+---------------------------------------------------------------------------------------------------+
| 10.42A PHYSICS-CONSTRAINED DEEPONET          | 10.42B FOURIER NEURAL OPERATOR (width 32, k=6)     |
|                                              |                                                    |
| [s_t, a_t] --> BRANCH MLP --+                | [s_t, a_t] on 14-sensor lattice x_i                |
|                             +-> <b, tau(y)>  |        | lift([u_i, x_i] -> 32ch)                  |
| aux grid y_q --> TRUNK MLP -+   + b0         |        v                                          |
|                             | = [dvx, dvy,   |  FFT -> keep 6 modes -> complex R_k -> IFFT       |
|                             v    contacts]   |    (+ pointwise bypass, GELU) x 2 blocks          |
|              HARD KINEMATIC SHELL (Section 4)|        | project -> scalar field f on [0, 1]      |
|              clamp v, X += vx/16, Y += vy/16 |        v                                          |
|              => zero residual BY CONSTRUCTION|  f(y_q) + b0[q] = hat_s_{t+1}[q]                  |
+---------------------------------------------------------------------------------------------------+
```

**Protocol.** Identical to Section 7 and Section 10.41: canonical 8,077-transition dataset, seed-42 episodic split, AdamW ($10^{-3}$ / $10^{-4}$), batch 128, Smooth L1, 35 epochs, identical 120-frame single-track and 10-start rollout evaluations. Each architecture's data pipeline is re-seeded and rebuilt inside its own loop iteration, so every row reproduces standalone under the canonical seed (the DeepONet reference row indeed re-produced the exact Section 10.41 numbers). Published comparison rows come from the committed `benchmark_metrics.json`.

#### 10.42.1 Empirical Results (seed 42, canonical split)

| Architecture | Paradigm | Parameters | Test Loss (Data MSE) | Kinematic Residual | Kin. Violations (single track) | Vel. Violations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| DeepONet (10.41 re-run) | Operator, unconstrained | 52,744 | 7.8800 | 273.15 | 118 / 120 (98.3%) | 0 / 120 |
| **Physics-Constrained DeepONet** | **Operator + hard kinematics** | 52,742 | **0.5766** | **0.0025** (analytical zero) | **0 / 120 (0.0%)** | **0 / 120** |
| **FNO (6 modes, width 32)** | **Spectral operator, unconstrained** | **14,537** | **0.4025** | 0.6579 | 80 / 120 (66.7%) | 36 / 120 (30.0%) |
| Hard Residual PINN (published) | Hard inductive bias | 36,486 | 0.5783 | 0.0019 | 0 / 120 (0.0%) | 0 / 120 |
| Statistical MLP (published) | Black-box | 36,744 | 16.4717 | 17,561.24 | 118 / 120 | 40 / 120 |

Multi-start aggregate (10 starts x 120 frames): Physics-Constrained DeepONet mean drift $164.12 \pm 142.52$ px with **0.0%** kinematic and velocity violation rates; FNO mean drift $210.79 \pm 175.50$ px with **88.4%** kinematic and **28.8%** velocity violation rates; DeepONet re-run $149.41 \pm 81.10$ px (99.6% kinematic violations). Single-track drifts were 102.84 px (PC-DeepONet, final 287.15 px) and 133.86 px (FNO, final 352.61 px), against the published Hard PINN's 50.87 px (final 78.13 px).

#### 10.42.2 What the Operator Family Study Establishes

1. **The hybrid works exactly as designed.** The Physics-Constrained DeepONet matches the Hard Residual PINN within 0.3% on single-step loss (0.5766 vs. 0.5783), with a kinematic residual at float32 zero (0.0025) and 0 violations on every rollout frame in the study. The Section 10.41 conclusion holds in both directions: the branch/trunk factorization is a viable force estimator, and hard kinematics transfer their guarantees to it unchanged. For planning under constraint-sensitivity (Sections 10.6, 10.19), the hybrid is a drop-in substitute class for the Hard PINN.
2. **FNO is the strongest physics-free single-step model in the repository - and still not a safe dynamics model.** With only 14,537 parameters (60% fewer than the published Hard PINN's 36,486) the spectral operator reaches a test loss of 0.4025, **30% below the Hard PINN** and 41x below the MLP: the globally smooth Fourier kernels fit the fixed-point position/velocity warps exceptionally well. Yet its residual (0.6579, 347x the analytical zero) is not structural: it violates the discrete identity on 88.4% of multi-start rollout frames and the engine's velocity bounds on 28.8%. This is the sharpest illustration yet of the benchmark's thesis - *single-step regression accuracy and discrete physical consistency are different objectives*, and only hard inductive bias delivers the second.
3. **Precision without guarantees is a planning hazard, not a planning asset.** FNO's rollout behavior (80/120 violated frames, 36 velocity violations on the single track, drift variability $\sigma = 175.50$ px) shows a model that is locally excellent but globally unconstrained: an MPC planner rolling it forward would trust sub-pixel-accurate steps that silently leave the engine's reachable set. Within the operator family the ordering for *world-model duty* is therefore PC-DeepONet $\gg$ DeepONet $\gt$ FNO, exactly perpendicular to the single-step ordering FNO $\gt$ PC-DeepONet $\gt$ DeepONet - the benchmark's central trade-off, now measured inside one architecture family.

Regenerate: `python -m src.evaluation.operator_benchmark` (emulator-free; `--fno-width`, `--fno-modes`, `--fno-layers` and `--latent-dim` control the architecture knobs; each row re-seeds independently).

---
