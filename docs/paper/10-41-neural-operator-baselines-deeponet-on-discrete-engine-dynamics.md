[← 10.40](10-40-physics-parameter-identification-the-inverse-problem.md) · [Contents](../../README.md#contents) · [10.42 →](10-42-physics-constrained-deeponet-and-the-fourier-neural-operator.md)

### 10.41 Neural-Operator Baselines: DeepONet on Discrete Engine Dynamics

Sections 5-8 benchmarked two families of learners: **statistical black-boxes** (MLP, LSTM), which approximate a function $\mathbb{R}^{14} \to \mathbb{R}^8$, and **physics-informed networks** (Soft/Hard PINN), which embed the discrete kinematics of Section 4. This section adds a third, qualitatively different family: **neural operators**. A DeepONet (Lu et al., 2021, *Nature Machine Intelligence* 3: 218-229) does not approximate a finite-dimensional map; it approximates an **operator $G: \mathcal{U} \to \mathcal{V}$ between function spaces**, with a universal approximation guarantee at the operator level. Introducing one answers a sharper question: *is the Hard PINN's advantage merely "more sophisticated architecture", or is it specifically the embedded discrete kinematics?*

**Formulation for the SMW engine map.** The one-step transition rule is recast as an operator between the *input function* $u$ (the instantaneous dynamic regime of the engine) and the *next-state field* $v$, as a sum of branch coefficients $b_k$ times trunk basis functions $\tau_k$:

$$v(y_q) = G(u)(y_q) = \sum_{k=1}^{p} b_k(u)\,\tau_k(y_q) + b_0[q],\qquad G(u)(y_q) \approx \hat s_{t+1}[q]$$

* **Branch net (sensors):** the input function is sampled at $m = 14$ fixed sensors - the combined state-action reading $u = [s_t, a_t] \in \mathbb{R}^{14}$ (each coordinate is one sensor of the current kinematic/contact/button regime). The branch MLP encodes it into $p = 64$ basis coefficients $b_k(u)$.
* **Trunk net (queries):** the trunk MLP evaluates $p$ basis functions $\tau_k$ at a query coordinate $y_q \in [-1, 1]$ that identifies the predicted state channel (the canonical grid spaces the 8 channels evenly between -1 and 1, stored as a module buffer). Because the trunk is a continuous function of $y$, the trained operator can also be queried at arbitrary intermediate coordinates.
* **Output assembly:** $\hat s_{t+1}[q] = \langle b(u), \tau(y_q) \rangle + b_0[q]$. The inner product imposes a **bilinear bottleneck of rank $p$**: the branch learns a dictionary of 64 global response patterns indexed by the input regime, the trunk learns how each pattern distributes over output channels - an inductive bias orthogonal to both the monolithic MLP (no factorization) and the Hard PINN (analytic integration).

```
+---------------------------------------------------------------------------------------------------+
|                        ARCHITECTURE 5: DEEPONET NEURAL OPERATOR BASELINE                          |
|                                                                                                   |
|  input function u = [s_t, a_t] (14 sensors)        query coordinate y_q (output channel)          |
|           |                                              |                                        |
|           v                                              v                                        |
|     BRANCH MLP (128x128) ---> b(u) in R^64      TRUNK MLP (128x128) ---> tau(y_q) in R^64        |
|           |                                              |                                        |
|           +----------------> inner product <b, tau> + b0[q] = hat_s_{t+1}[q]                      |
|                                                                                                   |
|  No analytic kinematics embedded: a pure operator-learning middle point between the               |
|  statistical MLP and the Hard Residual PINN.                                                      |
+---------------------------------------------------------------------------------------------------+
```

**Protocol.** Identical to the canonical benchmark (Section 7): the same 8,077-transition WRAM dataset, seed 42 episodic split (6,329 / 392 / 1,356), AdamW ($\eta = 10^{-3}$, weight decay $10^{-4}$), batch 128, Smooth L1 objective, `ReduceLROnPlateau`, early stopping (patience 8), 35 epochs maximum, 120-frame open-loop rollouts plus the 10-start multi-start variant. Implementation: `src/models/deeponet.py` (`DeepONetDynamics`, **52,744 trainable parameters** - 1.44x the MLP's 36,744), study orchestrator `src/evaluation/deeponet_benchmark.py`, metrics in `results/deeponet_benchmark_metrics.json`; comparison rows are read from the committed `results/benchmark_metrics.json`, not re-trained.

#### 10.41.1 Empirical Results under the Unified Protocol

| Architecture | Paradigm | Test Loss (Data MSE) | Kinematic Residual | Kinematic Violations (rollout) | Velocity Violations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Statistical MLP | Black-box function approximator | 16.4717 | 17,561.24 | 118 / 120 (98.3%) | 40 / 120 |
| Statistical LSTM | Recurrent black-box | 39.2194 | 37,361.16 | 120 / 120 (100%) | 0 / 120 |
| Soft-Constrained PINN | Loss-penalty physics | 53.8167 | 108,520.99 | 120 / 120 (100%) | 0 / 120 |
| **DeepONet** | **Neural operator (branch/trunk)** | **7.8800** | **273.15** | 118 / 120 (98.3%) | **0 / 120** |
| Hard Residual PINN | Hard inductive bias | **0.5783** | **0.0019** | **0 / 120 (0.0%)** | 0 / 120 |

Multi-start aggregate (10 starts x 120 frames): DeepONet mean drift $149.41 \pm 81.10$ px with a **99.58% kinematic-violation rate** and 0.0% velocity-violation rate; the published multi-start means are MLP 179.87 px (97.9% violations), LSTM 211.99 px (100%), Soft PINN 317.59 px (100%), Hard PINN 159.51 px (**0%** violations). On the single continuous 120-frame track DeepONet's mean drift was 43.10 px (final 117.80 px), against 50.87 px (final 78.13 px) for the Hard PINN - read with Section 10.4/10.6 caution: single-start drift is one draw, not an ordering guarantee.

#### 10.41.2 What the Operator Prior Buys - and What It Cannot Buy

1. **The operator factorization beats every black-box baseline of Section 7 - though 10.42 below finds a better unconstrained model.** DeepONet's test loss of 7.8800 is **2.1x lower than the statistical MLP** (16.4717), 5.0x lower than the LSTM and 6.8x lower than the Soft PINN, at 1.44x the MLP's parameter count and 11.08 s of training. Generalizing from sensor readings through a learned basis dictionary, rather than memorizing a monolithic map, measurably helps on this discrete system.
2. **But it does not discover the exact integration identity.** The kinematic residual drops from 17,561 (MLP) to 273 - a 64x improvement from inductive architecture alone - yet remains **143,000x above the Hard PINN's analytical zero** (0.0019), and autoregressive rollouts violate the discrete update identity on 99.6% of frames, statistically indistinguishable from the MLP's 98.3%. Universal approximation of the operator is not the same as satisfying $\Delta X = v_x/16.0$; that identity enters only through structure, not through architecture capacity or operator-level expressiveness.
3. **Drift and physical consistency decouple.** DeepONet posts the lowest non-physics drift figures of the study (single-start 43.10 px; multi-start 149.41 px) while violating kinematics on nearly every frame - trajectory-level coincidence without step-level consistency is exactly the failure mode a physics formulation is designed to exclude, and the reason this benchmark reports violation rates beside drift rather than drift alone.
4. **Position the family in the taxonomy.** DeepONet occupies the "structure-free but operator-aware" slot: meaningfully better than raw statistical learning, decisively below structural physics (13.6x the Hard PINN's error). The natural next step on this branch is a **physics-constrained neural operator** - a DeepONet whose trunk output is integrated through the Section 4 kinematics, or an FNO (Fourier Neural Operator) ablation over the same sensors - while Section 10.40 already resolves the constants such a hybrid would rely on.

Regenerate: `python -m src.evaluation.deeponet_benchmark` (emulator-free; `--latent-dim` controls the basis width $p$).

---
