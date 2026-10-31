[← 10.29](10-29-consolidation-of-frontiers-a-b-c-and.md) · [Contents](../../README.md#contents) · [10.31 →](10-31-end-to-end-pixel-perception-pixel-to-action-front-end.md)

### 10.30 Scope, Limitations & Threats to Validity

In adherence to rigorous scientific methodology, we explicitly delineate the boundary conditions, core assumptions, and threats to internal and external validity of the empirical findings presented in this study:

1. **Deterministic vs. Stochastic Transitions:**
   - *Super Mario World* is fundamentally a deterministic discrete-time dynamical system when conditioned on the exact WRAM microstate, hardware frame counter, and controller latch registers.
   - The reported near-zero prediction error of the Hard Residual PINN ($\text{MSE} = 0.58$) directly leverages this underlying engine determinism. In systems exhibiting non-negligible transition stochasticity (such as physical robotics with sensor noise or stochastic actuation delays), single-point residual models would require probabilistic formulations (e.g., Gaussian or categorical distributions, as benchmarked via our epistemic Deep Ensemble in Section 10.10).

2. **Privileged WRAM State Telemetry vs. Pixel-Level Vision:**
   - Both the world models and closed-loop control policies in this repository operate on privileged internal state vectors extracted directly from the SNES RAM bus (`$7E:0094`, `$7E:00D1`, `$7E:C800`).
   - This experimental design intentionally isolates the inductive physical biases from the confounding representations and optimization challenges of visual autoencoders or convolutional encoders. However, it means the current architecture is not an end-to-end pixel-to-action agent; it requires a state estimation front-end if deployed purely from visual observation streams.

3. **Structural Kinematic Priors vs. Classical Continuous Conservation Laws:**
   - The embedded physical constraints ($\Delta X = v_x / 16$ and saturation clamps $[v_{\min}, v_{\max}]$) are **discrete kinematic consistency identities** and **microprocessor arithmetic state invariants** arising from 16-bit fixed-point subpixel accumulator operations in the Ricoh 5A22 CPU.
   - They do not represent continuous Noether-derived conservation laws (such as continuous energy or momentum conservation in Hamiltonian mechanics). We explicitly maintain this terminology to prevent conflation between continuous analytical physics and discrete fixed-point computer arithmetic.

4. **Domain-Specific Inductive Bias:**
   - The hard kinematic residual layer assumes prior knowledge of the discrete time-step ($\Delta t = 1$ frame) and the subpixel scaling constant ($1/16$ px per subpixel unit).
   - While this structural formulation generalizes seamlessly across distinct game levels governed by the same engine routines (as demonstrated in Sections 10.16 and 10.28), porting to dynamical systems with unknown discretization schemes would necessitate either explicit system identification or meta-learning of the physical scaling factors. *(This is exactly the deferred capability realised in Section 10.40, where the unknown scaling constant and gravity are recovered by inverse-problem system identification and shown to enable zero-shot control transfer.)*

5. **Local Trajectory Optimization and Non-Convex Barriers:**
   - In zero-shot cross-stage navigation with complex multi-height obstacles (e.g., pipe structures or vertical walls), pure local trajectory optimization (such as standard CEM MPC without global topological pathfinding) can suffer from local minima and horizon truncation. Addressing this requires pairing local predictive models with amortized global policies (e.g., DAgger or Dyna-PPO) or multi-scale planning hierarchies.
