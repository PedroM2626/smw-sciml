[← 10.45](10-45-does-the-velocity-ceiling-become-measurable-under.md) · [Contents](../../README.md#contents) · [10.47 →](10-47-is-it-the-shell-or-the-parameterisation.md)

### 10.46 Do Learned Dynamics Models Contain the Engine's Constraints?

Sections 10.41 and 10.42 trained the neural-operator family and scored it the way this repository scores every model: single-step error, multi-start rollout drift, a physical-consistency rate. Sections 10.40 and 10.43 then asked a different question - can the constants and the constraints be *recovered* - and answered it with probes: the ceiling is the fixed point of the driven map, the gravity gate is a median tier separation. The two lines had never met. `src/evaluation/learned_structure_probe_benchmark.py` puts the committed checkpoints of Sections 8, 10.41 and 10.42 under the same interrogation, with the same acceptance rules, and asks whether a model that predicts well also *contains* the constraint.

One distinction decides whether a positive result means anything, and it is measured rather than assumed. The Hard Residual PINN and the Physics-Constrained DeepONet integrate their learned residual through a hard kinematic shell that clamps $v_x$ at a constructor constant - `max_vx = 72.0` in both. A fixed point in those two could be the architect's number rather than the network's, and the probe would be a tautology. It is not: the clamp sits at 72.0, the observed velocity support of the recording is 49.0, and the probed range never reaches the clamp - so `shell_clamp_binds_within_support` is false for both, and every crossing reported below is the network's own.

| model | clamp in architecture | max driven acceleration (px/frame) | stable ceiling it implies | gravity tier separation (true -2.80) | gate recovered |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Statistical MLP | no | 12.33 | 30.07 | -0.86 | no |
| Soft-PINN | no | 21.33 | 21.33 | -0.00 | no |
| Hard Residual PINN | yes (72.0, never binds) | 1.72 | 23.21 | -0.57 | no |
| DeepONet (10.41) | no | 13.21 | 35.61 | +0.28 | no |
| Physics-Constrained DeepONet (10.42) | yes (72.0, never binds) | 1.91 | 25.49 | -0.09 | no |
| FNO (10.42) | no | 1.61 | 21.68 | -0.18 | no |

1. **Every learned model has a plateau; only half of them have a ceiling.** All six reach a stable positive fixed point of the driven map - but three of them do it after accelerating at 12 to 21 px/frame, which is seven to twelve times the engine's largest real increment. A map that throws the velocity forward that hard and then folds back is not modelling a bound, it is leaving the data manifold, and the crossing is an artifact of exactly that. The study therefore classifies a crossing as a ceiling only if it is positive, inside the observed support, approached from below while the map still accelerates, stable from above, *and* reached with a plausible traction ($\leq 2.5$ px/frame). Three models pass (FNO 21.68, Hard PINN 23.21, PC-DeepONet 25.49); three are recorded as `crossing_without_plausible_traction`. This is a correction to the method, not just a result: the 10.43 acceptance rule, written for a genetic program that might collapse to a constant, is not sufficient for a float32 network - on the identity map it reports a bound at -0.5 purely from round-off, and `tests/test_learned_structure_probe.py` pins that behaviour down.
2. **None of the plateaus is the engine's ceiling.** The three plausible ones sit between 21.7 and 25.5, against a WRAM reference of 72.0, an observed support of 49.0, and the sustained flat-ground plateau that Section 10.45 measured on purpose-targeted telemetry at 36.075. So the answer to "does a well-fitting learned model contain the constraint" is *no, and not approximately*: what they contain is a plateau of their own making, in a region no recording ever visits.
3. **The gravity gate is absent from every one of them.** The largest tier separation any model shows is 0.86 px/frame against a true step of -2.80, so no model recovers the held-jump discontinuity. This agrees with the two families that were actually tested for it: genetic programming recovered it in 1 of 3 replicates and PySR in 3 of 3 only once its constants were numerically optimised (10.43.9), and no engine recovered it from real telemetry at all (10.43.9, 10.45). Across search, structure and learning, the gate is the hard one.
4. **One coincidence, recorded rather than buried.** The plain DeepONet's implied plateau, 35.61, is 1.3% from the console's measured sustained sprint ceiling of 36.075 (10.45). With the other five models sitting between 21 and 30, that agreement is not evidence of anything - the DeepONet's traction is also wildly wrong (13.2 px/frame) - but it is the kind of number a future study should test rather than smooth over, so it is in the artifact.

**Why not identify the constants *through* a learned model - and what happens when you try.** The obvious next question is whether a trained operator can serve as the forward model inside the 10.40 identification loop, differentiating end to end. The argument against is short: the repository already has the exact forward in closed form and differentiable (`simulate_step`, gradient verified w.r.t. $\theta$), so a learned surrogate cannot remove any error from the identification - it can only add its own. And the other direction, differentiating through the *console*, is unavailable for a structural reason: `src/environment/snes_emulator.py` is a ctypes binding with no tensor path at all, which is why every emulator-in-the-loop method in this repository (MPC, Dyna-PPO, DAgger) is gradient-free by necessity. Rather than leave that as prose, the study measures it: identify the seven constants twice, once against the real transitions and once against pseudo-transitions rolled out by each model, with the same warm start, the same 900 Adam steps and the same seed.

| identification against | worst-case relative error vs the WRAM reference | largest shift of any constant from the direct fit |
| :--- | :---: | :---: |
| the real recording (the published 10.40-E2 arm) | 84.1% | - |
| FNO pseudo-transitions | 87.2% | 2.25 |
| Hard Residual PINN | 99.0% | 3.85 |
| Physics-Constrained DeepONet | 99.2% | 3.93 |
| Statistical MLP | 148.1% | 18.29 |
| DeepONet | 527.8% | 17.51 |
| Soft-PINN | 3933.0% | 23.79 |

The direct fit's 84.1% reproduces the number Section 10.40-E2 publishes, which is the cross-check that the harness is the same one. Every surrogate is worse than the data it was trained on, by an amount that tracks its own quality (the two kinematically-constrained models cost 3.9 units of drift, the Soft-PINN costs 23.8 and its identification loss diverges to $4.8\times10^{7}$ because the analytic map simply cannot fit its trajectories). That is the argument settled empirically: identification through a surrogate recovers the surrogate's bias, expressed in physical units of the constants, and buys nothing - because the thing a surrogate is for, an expensive or non-differentiable forward model, is not what this problem has.

![Structural probes on learned dynamics models](../../results/figures/learned_structure_probe.png)

*Left: which models have a fixed point under the 10.43 rule, and which keep one after the traction test (orange = the clamp the architecture wrote, otherwise the network's own). Right: the maximum acceleration each model produces when fully driven - the reason three of the six crossings are artifacts.*

**Limitations.** The probes interrogate the velocity law in isolation: contact channels are held at free flight and the action is fixed at sprint-right, so a model that implements a ceiling only in combination with terrain context would be missed - which is the same restriction the 10.43 probes have, chosen deliberately so the numbers are comparable. Each checkpoint is the single published fit, so the probe inherits training variance it cannot see. The 2.5 px/frame traction threshold is a chosen constant, and both the raw crossing and the gain are reported so a reader can re-classify without re-running anything. The surrogate arm compares *data sources* under identical optimisation; it is not a tuning study, and a better-optimised identification through a surrogate would still be bounded by that surrogate's bias.

Regenerate: `python -m src.evaluation.learned_structure_probe_benchmark` (emulator-free, ~1 min CPU; `make learned-probes`; `--id-steps` controls the identification arms and `--probe-rows` the resolution of the driven-map scan).
