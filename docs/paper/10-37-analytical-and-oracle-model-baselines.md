[← 10.36](10-36-yoshi-s-island-2-capture-blocked-with.md) · [Contents](../../README.md#contents) · [10.38 →](10-38-closed-loop-reproduction-audit-and-preamble-probe.md)

### 10.37 Analytical and Oracle-Model Baselines

Every result above compares learned models against each other. Two reference
points were missing: how much of the Hard PINN's advantage is the physics prior
itself, and what a near-exact model would achieve in closed loop.
`src/evaluation/analytical_baselines.py` supplies both.

#### 10.37.1 Baseline A - Engine Rules with Zero Hidden Units

`AnalyticalKinematicsDynamics` (`src/models/analytical_kinematics.py`) contains no
neural network at all: exact fixed-point integration $X_{t+1} = X_t + v_x/16$, the
documented saturation tiers (walk 20 / run 48 / sprint 72, $v_y \in [-80, 64]$),
asymmetric gravity (+3 held / +6 falling) and six interpretable scalars
(walk/run traction, friction, skid deceleration, jump impulse and its momentum
gain) identified by deterministic coordinate-wise grid search on the same train
split, same seed and same loss as the neural baselines.

| Channel (1,356 test transitions) | Engine rules (0 learned parameters) | Hard PINN (36,486 parameters) |
| :--- | :---: | :---: |
| $X$ MSE / R² | **0.1397** / 1.0000 | - |
| $Y$ MSE / R² | 4.6748 / 0.9960 | - |
| $v_x$ MSE / R² | **3.2078** / 0.9900 | - |
| $v_y$ MSE / R² | 931.03 / 0.4864 | - |
| `c_ground` accuracy / F1 | 0.9904 / 0.9938 | - |
| All-channel test MSE | 117.38 | **0.5783** |
| Kinematic residual $\|\Delta X - v_x/16\|^2$ | **exactly 0.0** | 0.0019 |

**The horizontal channel is fully explained by the published rules** - position MSE
0.14 px and velocity MSE 3.21 (subpixels/frame)² with no hidden units, no
gradients and 6 interpretable scalars. That is the ceiling any dynamics model should
be measured against, and it localises where learning is actually needed: the jump
impulse and the contact flags.

Two honest caveats are reported instead of being fitted away:

1. **The scalars are only weakly identifiable.** Grid search pinned walk traction,
   run traction and friction all at the bottom of their ranges (0.50) and the
   impulse at the boundary -64.0, while the velocity MSE improved only from
   515.99 to 461.41 across all six parameters - the landscape is nearly flat in the
   horizontal scalars once the vertical error dominates.
2. **The jump impulse is not recoverable from single transitions.**
   `vertical_dynamics_diagnostics` shows that 94.7% of frames flagged as take-off
   are *already rising*, only 21.1% have the jump button held, and observed
   post-impulse velocity reaches -112 subpixels/frame - outside the documented
   $[-64, -80]$ window of §4.2.1. Held-ascent $\Delta v_y$ is reproduced exactly
   (mean +3.00, std 0.00). Conclusion: WRAM stores velocity *after* the engine's
   impulse, so a transition dataset observes the integral, not the jump. Recovering
   it needs the input-latch edge at `$7E:0016` or a multi-frame window.

#### 10.37.2 Baseline B - Oracle-Model MPC Upper Bound

Same savestate, same objective (weights copied verbatim from the §10.6 protocol),
same CEM budget ($H = 15$, 256 candidates, 3 iterations), same seed - only the
dynamics the planner rolls out differ
(`results/oracle_mpc_metrics.json`):

| Controller | Progress (300 frames) | Survived | Control throughput |
| :--- | :---: | :---: | :---: |
| MPC + Hard Residual PINN (learned, 36,486 parameters) | 573.94 px | 300 / 300 | 40.3 FPS |
| MPC + Analytical Engine Rules (0 learned parameters) | **605.00 px** | 300 / 300 | 17.0 FPS |

Action agreement between the two planners: 0.0% on the first commanded action, 43.3%
over the full 300-frame sequence.

Reading and limits of the claim: in closed loop the parameter-free rules are **not**
beaten by the learned model (+5.4% progress) even though their all-channel
single-step MSE is 200x worse - MPC only needs an approximately correct
short-horizon gradient, and a model that cannot violate the kinematics never has to
learn not to. This bounds what §10.6-§10.7 can attribute to *learning*: the gap from
a near-exact model to the best learned one is small for this objective, whereas the
gap from an unstructured one is large (66 px for the MLP, §10.6). Throughput halves
(17.0 vs 40.3 FPS) because the analytical forward pass is branch-heavy tensor code
rather than one dense matmul - interpretability is not free.

---
