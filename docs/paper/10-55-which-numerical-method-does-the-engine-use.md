[← 10.54](10-54-auditing-the-physics-prose-of-section-4.md) · [Contents](../../README.md#contents) · [10.56 →](10-56-canonical-sparse-identification-on-every-recording-this.md)

### 10.55 Which Numerical Method Does the Engine Use? The Solver as the Axis

§10.53 settled *which velocity* advances position and left the question underneath it untouched. If position is an integral of velocity, a model can get the velocity right and still integrate it with the wrong method - and nothing in this repository has ever made the method a variable. Every dynamics model here, from the shells of 10.27 to the operators of 10.42 to the convention arms of 10.53, takes exactly one state-to-state step per frame, so the numerical method is implicit in the code rather than an axis you can move. `src/models/neural_ode_dynamics.py` makes it one: the same network is read as a *rate* instead of a difference, $\mathrm{d}x/\mathrm{d}t = v_x/16$ and $\mathrm{d}v/\mathrm{d}t = a(s,a)$, and one frame is one integration step of that field at $\Delta t = 1$ under one of four named methods - forward Euler, the velocity-first (semi-implicit, "symplectic") split that the published shells implement, explicit midpoint, and classical RK4. Three families x four solvers x two mechanisms, five seeds, the protocol of 10.47/10.53 unchanged.

Two properties make the axis measurable rather than decorative. First, the four methods differ in exactly one term: written as the §4.1 identity plus the correction, $\hat x = x + v_t/16 + c\,\hat a/16$, the coefficient is $c = 0$ for Euler, $c = 1$ for the velocity-first split and $c = \tfrac{1}{2}$ for midpoint and RK4, which is what the exact solution of a constant-acceleration step would give. Second, $c$ is not read off that algebra - the study *fits* it from each trained arm's held-out behaviour (least squares through the origin in $\hat a$) and applies the identical estimator to the recorded console transitions. "The engine is forward Euler" then stops being a code-reading claim and becomes a measured slope in the same units as the arms', with the same estimator on both sides. Unconstrained arms:

| solver | family | position error (px) | effective-velocity error (sub-px) | measured $c$ | $c$ the method predicts | drift (px) | viol @0.2 px | viol @0.002 px | frames the step matches the recording |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `euler` | MLP | 0.1056 | 1.69 | 0.0000 | 0 | 160.4 | 0.0000 | 0.0000 | 93.5% |
| `euler` | DeepONet | 0.1056 | 1.69 | 0.0000 | 0 | 159.0 | 0.0000 | 0.0000 | 93.5% |
| `euler` | FNO | 0.1056 | 1.69 | 0.0000 | 0 | 149.0 | 0.0000 | 0.0000 | 93.5% |
| `symplectic` | MLP | 0.1359 | 2.17 | 1.0000 | 1 | 167.1 | 0.0000 | 0.9346 | 52.9% |
| `symplectic` | DeepONet | 0.1366 | 2.19 | 1.0000 | 1 | 161.5 | 0.0012 | 0.9575 | 52.4% |
| `symplectic` | FNO | 0.1281 | 2.05 | 1.0000 | 1 | 138.6 | 0.0084 | 0.9484 | 71.1% |
| `midpoint` | MLP | 0.1209 | 1.93 | 0.5025 | 0.5 | 160.9 | 0.0000 | 0.8545 | 80.9% |
| `midpoint` | DeepONet | 0.1209 | 1.93 | 0.5042 | 0.5 | 153.5 | 0.0000 | 0.8807 | 80.4% |
| `midpoint` | FNO | 0.1185 | 1.90 | 0.5039 | 0.5 | 153.7 | 0.0000 | 0.9345 | 87.4% |
| `rk4` | MLP | 0.1208 | 1.93 | 0.5009 | 0.5 | 161.1 | 0.0000 | 0.8582 | 81.0% |
| `rk4` | DeepONet | 0.1199 | 1.92 | 0.5012 | 0.5 | 165.2 | 0.0000 | 0.9227 | 81.0% |
| `rk4` | FNO | 0.1182 | 1.89 | 0.5014 | 0.5 | 155.7 | 0.0022 | 0.9467 | 87.6% |
| recorded console | - | - | - | 0.1912 | 0 | - | 0.0462 | 0.0462 | 93.5% |

Projecting the Section 4 velocity window *inside* the step (clamp the derived velocity before it advances anything) against leaving the field free, averaged over families:

| solver | bounded position (px) | bounded drift (px) | bounded out-of-bounds | bounded max $\|\hat v_x\|$ | free position (px) | free drift (px) | free out-of-bounds | free max $\|\hat v_x\|$ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `euler` | 0.1056 | 139.7 | 0.000 | 42.04 | 0.1056 | 156.1 | 0.188 | 41.90 |
| `symplectic` | 0.1327 | 135.5 | 0.000 | 41.89 | 0.1335 | 155.7 | 0.188 | 41.88 |
| `midpoint` | 0.1189 | 132.5 | 0.000 | 42.09 | 0.1201 | 156.0 | 0.188 | 41.81 |
| `rk4` | 0.1194 | 142.4 | 0.000 | 42.06 | 0.1196 | 160.7 | 0.188 | 41.84 |

1. **Every arm measures the second-order coefficient its own method prescribes, to four decimals.** Euler 0.0000, the velocity-first split 1.0000, midpoint 0.5025-0.5042, RK4 0.5009-0.5014 across three families and five seeds. The solver is therefore a real axis in this repository rather than a rename of the residual head: the same weights read under a different integrator give a different map, and the fitted coefficient behaves exactly as the truncation-error analysis says it must - including the way midpoint's overshoot (0.5042 for the DeepONet) exceeds RK4's (0.5012) by an order of magnitude in error terms, which is the method's order showing up in a measured quantity.
2. **The console's fitted coefficient is 0.4956 horizontally (0.1912 pooled) - and that number is an artifact of a handful of frames.** Read like a slope it says the engine is a second-order method, which contradicts what §10.49 and §10.53 measured. Read like a distribution it says something better: over the five held-out splits the console satisfies $x + v_{x,t}/16$ exactly (within half a sub-pixel) on 93.5% of frames, and the frames that carry the whole slope are an average of 12.6 per split out of 1,792, with a mean residual of -17.73 sub-pixels. Restricted to the frames where no contact flag is set, the fitted slope falls to 0.0720 and 99.3% of them are exact. So the exceptions are collisions - wall stops, screen wraps, ridden platforms, terminations - and away from them the engine's step is a first-order accumulation with no acceleration term at all. §4.1's identity is not an approximation the engine refines; it is the whole rule, and the least-squares slope over every frame was measuring the frames the rule does not cover.
3. **The method with no second-order term wins the prediction metrics, because the target has no second-order term.** Euler's held-out position error is 0.1056 px in all three families - the same number 10.53 published for its `carried` cells, and for the same structural reason: position is a deterministic function of the input state, so no weight vector can move it. Midpoint and RK4 pay 0.1182-0.1209 px, the velocity-first split 0.1281-0.1366 px. On the predicate ladder the gap is decisive rather than marginal: the Euler arms hold the published violation rate at 0.0000 down to 0.002 px, while midpoint and RK4 are flagged on 0.8545-0.9467 of their rollout frames there and the split on 0.9346-0.9575 - and all of them at 0.0000-0.0084 at the tolerance the repository has always published. That is 10.53's finding 2 with its mechanism attached: the shells' zeros at 0.2 px are a half-step of acceleration, and this study can name the half-step, because $c$ is exactly the quantity that separates them from Euler.
4. **Higher-order accuracy is not just unhelpful, it is aimed at the wrong error.** Midpoint and RK4 evaluate the learned field twice and four times per frame respectively, which is the whole point of them: they are built to reduce the truncation error of integrating a *smooth* flow. Here the field is discontinuous across a contact boundary and the contacts are read once, from the state the frame starts from, so a mid-step evaluation cannot see the collision that ends the step - the method's extra work is spent interpolating a vector field that the engine never interpolates. Their drift is no better than Euler's (153.5-165.2 px against 149.0-160.4) and their position error is worse, which is what one expects when the exact solution of the posited ODE is not what the console computes.
5. **In-step projection does its job in every solver, and is invisible to Euler's position channel.** Out-of-bounds goes 0.188 to 0.000 in all four methods, and the clamped velocity never exceeds the run cap (max $|\hat v_x|$ 41.89-42.09 against the cap of 48, i.e. neither the free nor the bounded arm is being clipped by the wall it is allowed to reach). The Euler row makes the invariance visible: bounding changes its drift (156.1 to 139.7 px) but cannot change its position error, which stays at exactly 0.1056 px, because in that method the projected velocity never enters the position update at all - the step is $x + v_t/16$ and $v_t$ is the input.
6. **The control side of the design does not reproduce, and the size of the failure is a finding.** Euler is 10.53's `carried` cell and the velocity-first split is its `next` cell, with the same weights, same data, same seeds; the two code paths differ in that this model reads the network as an acceleration and evaluates it more than once per frame. Position error reproduces to 0.0000 px (both sides are structural), but multi-start drift does not: 0.0008 px of gap for `MLP/euler` against `MLP/carried/none` and 0.0075 px for the MLP's split against `next`, 14.14 and 4.97 px for the DeepONet, and 34.79 and 88.00 px for the FNO. That gap is attributable to the second parameterisation and not to the machine, because the command was run twice for this section and the twelve published checkpoints came back byte-for-byte identical - `results/checkpoints/ode_published.sha256` records the first run's hashes and a test recomputes the committed files against them. The clamped cells, which publish no weights and are therefore not covered by that check, did move between the same two runs, which is the second reason this section quotes drift and rests nothing on it. A cross-artifact difference of up to 88 px is larger than every drift effect 10.47 and 10.50 ranked arms by; the position and predicate cells, where the gaps are structural zeros, are the ones that carry the findings.
7. **Closed loop, the solver is worth more than the mechanism was in 10.47 - for two of three families, and not for the third.** Same planner, objective, savestate, 300-frame budget and five CEM seeds as 10.44/10.47/10.53, flying the twelve unconstrained arms beside the same published references:

| controller | progress (px) | worst seed | best seed | frames | deaths | vs the established rules |
|:--|--:|--:|--:|--:|--:|:--|
| `established_wram_engine_rules` | 619.90 ± 20.02 | 601.50 | 643.38 | 300.0 | 0 | reference |
| `published_pc_deeponet_10_42` | 635.61 ± 14.18 | 610.25 | 642.31 | 300.0 | 0 | +15.71 px, $d_z$ +0.50, $p$ 0.325 |
| `published_hard_pinn_10_27` | 299.20 ± 253.32 | 113.56 | 579.44 | 224.6 | 3 | -320.70 px, $d_z$ -1.21, $p$ 0.054 |
| `mlp_euler_free` | 521.66 ± 229.09 | 113.12 | 637.00 | 274.8 | 1 | -98.24 px, $d_z$ -0.41, $p$ 0.416 |
| `mlp_symplectic_free` | 507.00 ± 221.13 | 114.44 | 633.38 | 274.6 | 1 | -112.90 px, $d_z$ -0.48, $p$ 0.345 |
| `mlp_midpoint_free` | 610.40 ± 22.61 | 591.69 | 635.19 | 300.0 | 0 | -9.50 px, $d_z$ -0.35, $p$ 0.474 |
| `mlp_rk4_free` | 614.09 ± 20.54 | 584.50 | 630.56 | 300.0 | 0 | -5.82 px, $d_z$ -0.18, $p$ 0.701 |
| `deeponet_euler_free` | 643.02 ± 0.08 | 642.88 | 643.06 | 300.0 | 0 | +23.12 px, $d_z$ +1.16, $p$ 0.061 |
| `deeponet_symplectic_free` | 642.38 ± 0.79 | 641.19 | 643.06 | 300.0 | 0 | +22.47 px, $d_z$ +1.08, $p$ 0.073 |
| `deeponet_midpoint_free` | 636.61 ± 12.58 | 614.12 | 642.62 | 300.0 | 0 | +16.71 px, $d_z$ +0.56, $p$ 0.278 |
| `deeponet_rk4_free` | 627.84 ± 20.40 | 601.38 | 643.00 | 300.0 | 0 | +7.94 px, $d_z$ +0.24, $p$ 0.614 |
| `fno_euler_free` | -0.69 ± 0.08 | -0.75 | -0.56 | 300.0 | 0 | -620.59 px, $d_z$ -31.08, $p$ 0.000 |
| `fno_symplectic_free` | 114.27 ± 1.17 | 113.44 | 116.31 | 173.0 | 5 | -505.63 px, $d_z$ -24.71, $p$ 0.000 |
| `fno_midpoint_free` | 307.92 ± 274.08 | 105.88 | 608.19 | 232.6 | 3 | -311.98 px, $d_z$ -1.09, $p$ 0.072 |
| `fno_rk4_free` | 59.87 ± 40.13 | 18.25 | 113.12 | 255.8 | 3 | -560.03 px, $d_z$ -12.57, $p$ 0.000 |

   `deeponet_euler_free` covers 643.02 ± 0.08 px - a between-seed spread of 0.08 px against the hand-written engine rules' own 20.02 px, the tightest this repository has recorded for a controller that actually progresses (the only tighter row anywhere is `fno_euler_free` at 0.078 px, which advances -0.69 px, so it is a tie with a stationary arm). All four DeepONet solvers land within 15.19 px of each other (the paired contrasts below), so for that family the solver is worth almost nothing and the open-loop ordering (Euler best at 0.1056 px) does not survive into control at all. For the MLP the ordering *inverts*: the two arms that die once and spread over 220 px are Euler and the split (521.66 and 507.00 px), while midpoint and RK4 reach the frame budget in all five seeds with a 20-23 px spread (610.40 and 614.09 px), i.e. +88.74 and +92.42 px over Euler at $p$ 0.42-0.44. The FNO is a lottery here as it was in 10.53, but a different draw: Euler makes no progress at all (-0.69 px), the split dies in all five seeds (114.27 px), and it is midpoint that reaches 307.92 px.

* `mlp_euler_free -> mlp_symplectic_free`: -14.66 px ($d_z$ -0.36, $p$ 0.472)
* `mlp_euler_free -> mlp_midpoint_free`: +88.74 px ($d_z$ +0.40, $p$ 0.417)
* `mlp_euler_free -> mlp_rk4_free`: +92.42 px ($d_z$ +0.39, $p$ 0.436)
* `mlp_symplectic_free -> mlp_midpoint_free`: +103.40 px ($d_z$ +0.48, $p$ 0.343)
* `mlp_symplectic_free -> mlp_rk4_free`: +107.08 px ($d_z$ +0.47, $p$ 0.356)
* `mlp_midpoint_free -> mlp_rk4_free`: +3.68 px ($d_z$ +0.10, $p$ 0.834)
* `deeponet_euler_free -> deeponet_symplectic_free`: -0.65 px ($d_z$ -0.78, $p$ 0.156)
* `deeponet_euler_free -> deeponet_midpoint_free`: -6.41 px ($d_z$ -0.51, $p$ 0.319)
* `deeponet_euler_free -> deeponet_rk4_free`: -15.19 px ($d_z$ -0.74, $p$ 0.172)
* `deeponet_symplectic_free -> deeponet_midpoint_free`: -5.77 px ($d_z$ -0.47, $p$ 0.356)
* `deeponet_symplectic_free -> deeponet_rk4_free`: -14.54 px ($d_z$ -0.71, $p$ 0.186)
* `deeponet_midpoint_free -> deeponet_rk4_free`: -8.77 px ($d_z$ -0.63, $p$ 0.229)
* `fno_euler_free -> fno_symplectic_free`: +114.96 px ($d_z$ +95.03, $p$ 0.000)
* `fno_euler_free -> fno_midpoint_free`: +308.61 px ($d_z$ +1.13, $p$ 0.066)
* `fno_euler_free -> fno_rk4_free`: +60.56 px ($d_z$ +1.51, $p$ 0.028)
* `fno_symplectic_free -> fno_midpoint_free`: +193.65 px ($d_z$ +0.71, $p$ 0.188)
* `fno_symplectic_free -> fno_rk4_free`: -54.40 px ($d_z$ -1.33, $p$ 0.041)
* `fno_midpoint_free -> fno_rk4_free`: -248.05 px ($d_z$ -0.85, $p$ 0.130)

   Eighteen paired contrasts, same family and same planner, only the integrator changing: three reach $p \lt 0.05$ and all three are FNO rows - `fno_euler_free -> fno_symplectic_free` (+114.96 px, $p$ 0.000, which is the comparison of an arm pinned at the start line with an arm that dies in every seed, not a win for either), `fno_euler_free -> fno_rk4_free` (+60.56 px, $p$ 0.028) and `fno_symplectic_free -> fno_rk4_free` (-54.40 px, $p$ 0.041). Every MLP and DeepONet contrast is individually insignificant at five seeds, as 10.53's were. No solver is a controller for one family and a disaster for another in a way this design can separate; what it does separate is the families.
8. **Reading 10.53's closed-loop result through this second code path does not reproduce it, and the gap is where the headline was.** The `euler` and `symplectic` arms are 10.53's `carried` and `next` cells, same family, same convention, same five CEM seeds - so their differences measure the reproducibility of the closed loop itself:

| family | 10.53 | progress (px) | deaths | 10.55 | progress (px) | deaths | difference |
|:--|:--|--:|--:|:--|--:|--:|--:|
| mlp | 10.53 `mlp_carried_none` | 521.01 ± 228.70 | 1 | 10.55 `mlp_euler_free` | 521.66 ± 229.09 | 1 | +0.65 px |
| mlp | 10.53 `mlp_next_none` | 498.84 ± 216.13 | 1 | 10.55 `mlp_symplectic_free` | 507.00 ± 221.13 | 1 | +8.16 px |
| deeponet | 10.53 `deeponet_carried_none` | 637.46 ± 11.62 | 0 | 10.55 `deeponet_euler_free` | 643.02 ± 0.08 | 0 | +5.56 px |
| deeponet | 10.53 `deeponet_next_none` | 623.06 ± 17.55 | 0 | 10.55 `deeponet_symplectic_free` | 642.38 ± 0.79 | 0 | +19.31 px |
| fno | 10.53 `fno_carried_none` | 468.69 ± 212.45 | 1 | 10.55 `fno_euler_free` | -0.69 ± 0.08 | 0 | -469.38 px |
| fno | 10.53 `fno_next_none` | -4.25 ± 2.98 | 0 | 10.55 `fno_symplectic_free` | 114.27 ± 1.17 | 5 | +118.52 px |

   Two of the three families reproduce to within 0.65-19.31 px, which is the resolution of this table. The FNO does not: the cell that 10.53 published as "the convention is the difference between a controller and none" (-4.25 px under `next`, 468.69 px under `carried`, +472.94 px, $t$-test $p = 0.007$) lands at -0.69 px when the same convention is reached through the ODE parameterisation, and its `next` twin moves from -4.25 to 114.27 px. Both FNO rows are therefore near the floor on one path and not on the other, and the honest statement is the weaker one: for the FNO the closed loop is dominated by which weights training happens to return, and a 472 px gain attributed to the velocity convention is inside the spread of that lottery. This is recorded as a correction in Section 12, not applied silently to 10.53 - the published row is what its own artifact measured, and it stands as a measurement of that checkpoint.

**Limitations.** A one-frame step of a learned field is not a NeuralODE in the usual sense: there is no adaptive stepping and no adjoint gradient - the field is trained one-step, exactly as the residual heads of 10.47 are - so this study measures *which one-step method the console is*, and any claim about the accuracy of RK4 on a smooth flow is out of scope and in fact contradicted by finding 4. Contacts are read once, from the state the frame starts from, which is what makes the field discontinuous inside a step and is the mechanism behind finding 4; a hybrid or event-detection integrator would be the honest way to test whether that is a property of the engine or of this protocol, and it is not implemented. The fitted $c$ is a least-squares slope and is dominated by the tail, which is why every $c$ in this section is quoted next to an exact-frame count; the console's contact-free subset averages 614 frames per split and its non-exact frames are a mean of 0.4, so that statistic is thin. Position error in the Euler arms is pinned at 0.1056 px by construction and says nothing about the network, as 10.53's finding 1 established; drift is quoted but not rested on, for the reason finding 6 measured; and the closed-loop table has five seeds, which puts a floor of $p \ge 0.0625$ on any single contrast.

![Which integrator advances the frame](../../results/figures/neural_ode_integrator.png)

*Left:* held-out position error per unconstrained arm on a log scale, with the console's own carried-convention residual as the dashed line - the three Euler bars sit exactly on it, and the other nine sit above it in order of their $c$. Right: the fitted second-order coefficient per solver against forward Euler's 0, the exact-continuous 1/2 and the velocity-first split's 1, with the console's own fitted pooled slope as the dotted line: the arms land on their methods, and the console's slope is the number finding 2 explains away.

Regenerate: `python -m src.evaluation.neural_ode_integrator_benchmark --seeds 42,43,44,45,46` (emulator-free, ~45 min on the RTX 4070 for 24 arms x 5 seeds; `--arms MLP` cuts it to a third, smoke config `configs/smoke_neural_ode.yaml`), then `python -m src.evaluation.physics_injection_mpc_benchmark --study ode` (hardware, ~35 min, writes `results/neural_ode_integrator_mpc_metrics.json`), or `make neural-ode` for both.
