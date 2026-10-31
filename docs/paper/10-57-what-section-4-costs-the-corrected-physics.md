[← 10.56](10-56-canonical-sparse-identification-on-every-recording-this.md) · [Contents](../../README.md#contents) · [10.58 →](10-58-the-residue-of-section-4-1-is.md)

### 10.57 What Section 4 Costs: the Corrected Physics, Measured One Line at a Time

Section 10.54 audits §4 and finds four claims whose own sections name a file that implements
something else. Those sentences have been rewritten to the telemetry; the code has kept its
published behaviour, because every artifact in this repository is a measurement made with it. The
corrections are now constructible - `position_velocity="carried"` on the shells and the closed-form
rules, `contact_rule="zero_increment"` on the penalty, and the rollout predicate scoring whichever
velocity bound the caller names. This section prices them: the same objects as 10.47's cells, built
by the same `build_arm`/`build_loss`, five seeds 42-46, 35 epochs, one axis changed at a time.

**Two rulers and two tolerances, per trajectory.** The published predicate (0.2 px, 72.0 and
64.0) is applied first, then the same rollouts against the bound 10.49 establishes as reachable
(48.0) and against a 0.002 px tolerance. **Same-weights replay** trains one arm and then scores
it under the other convention by flipping the attribute, which removes training noise from the
contrast entirely. **Parity control:** the default arms must reproduce 10.47's published cells,
because if they moved, the flags would have reached the path the paper stands on.

| arm | test loss | $x$ MAE (px) | drift (px) | kin. viol @0.2 | kin. viol @0.002 | vel. viol @72 | vel. viol @48 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Hard PINN/next | 0.6214 | 0.1348 | 137.06 | 0.0000 | 0.9158 | 0.0000 | 0.0433 |
| Hard PINN/carried | 0.6322 | 0.1056 | 135.86 | 0.0000 | 0.0000 | 0.0000 | 0.0313 |
| MLP/residual/hard/next | 0.6214 | 0.1348 | 137.06 | 0.0000 | 0.9158 | 0.0000 | 0.0433 |
| MLP/residual/hard/carried | 0.6322 | 0.1056 | 135.86 | 0.0000 | 0.0000 | 0.0000 | 0.0313 |
| DeepONet/residual/hard/next | 0.6257 | 0.1361 | 127.81 | 0.0000 | 0.9028 | 0.0000 | 0.0077 |
| DeepONet/residual/hard/carried | 0.6346 | 0.1056 | 137.31 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FNO/residual/hard/next | 0.6273 | 0.1295 | 136.69 | 0.0000 | 0.9593 | 0.0000 | 0.0188 |
| FNO/residual/hard/carried | 0.6385 | 0.1056 | 143.71 | 0.0000 | 0.0000 | 0.0000 | 0.0490 |
| engine rules/next | - | 0.1898 | 114.90 | 0.0490 | 0.4103 | 0.0000 | 0.1510 |
| engine rules/carried | - | 0.1056 | 113.27 | 0.0000 | 0.0000 | 0.0000 | 0.1510 |

| arm (state + soft) | median $\lvert v_y \rvert$ on grounded frames | median $\lvert \Delta v_y \rvert$ | rate $v_y = 0$ | test loss | drift (px) |
|:--|:--:|:--:|:--:|:--:|:--:|
| MLP/state/soft/published | 0.025 | 5.997 | 0.00% | 44.0360 | 264.50 |
| MLP/state/soft/increment | 2.930 | 5.990 | 0.00% | 43.5781 | 265.25 |
| DeepONet/state/soft/published | 1.086 | 5.810 | 0.00% | 21.6065 | 444.69 |
| DeepONet/state/soft/increment | 6.437 | 6.869 | 0.00% | 27.6039 | 456.49 |
| FNO/state/soft/published | 0.150 | 6.064 | 0.00% | 5.9590 | 103.23 |
| FNO/state/soft/increment | 6.895 | 1.658 | 0.00% | 2.4732 | 256.05 |

| arm | violation at 72.0 | at 48.0 | difference |
|:--|:--:|:--:|:--:|
| Hard PINN/next | 0.0000 | 0.0433 | +0.0433 |
| Hard PINN/carried | 0.0000 | 0.0313 | +0.0313 |
| MLP/residual/hard/next | 0.0000 | 0.0433 | +0.0433 |
| MLP/residual/hard/carried | 0.0000 | 0.0313 | +0.0313 |
| DeepONet/residual/hard/next | 0.0000 | 0.0077 | +0.0077 |
| DeepONet/residual/hard/carried | 0.0000 | 0.0000 | +0.0000 |
| FNO/residual/hard/next | 0.0000 | 0.0188 | +0.0188 |
| FNO/residual/hard/carried | 0.0000 | 0.0490 | +0.0490 |
| MLP/state/soft/published | 0.0000 | 0.0000 | +0.0000 |
| MLP/state/soft/increment | 0.0000 | 0.0000 | +0.0000 |
| DeepONet/state/soft/published | 0.0013 | 0.8733 | +0.8720 |
| DeepONet/state/soft/increment | 0.1880 | 0.3713 | +0.1833 |
| FNO/state/soft/published | 0.0000 | 0.0000 | +0.0000 |
| FNO/state/soft/increment | 0.1467 | 0.1913 | +0.0447 |
| engine rules/next | 0.0000 | 0.1510 | +0.1510 |
| engine rules/carried | 0.0000 | 0.1510 | +0.1510 |

| arm as trained | drift as trained | drift replayed the other way | replayed | difference |
|:--|:--:|:--:|:--:|:--:|
| Hard PINN/next | next | carried | 137.06 | 162.76 | +25.70 |
| Hard PINN/carried | carried | next | 135.86 | 170.09 | +34.23 |
| MLP/residual/hard/next | next | carried | 137.06 | 162.76 | +25.70 |
| MLP/residual/hard/carried | carried | next | 135.86 | 170.09 | +34.23 |
| DeepONet/residual/hard/next | next | carried | 127.81 | 163.32 | +35.51 |
| DeepONet/residual/hard/carried | carried | next | 137.31 | 169.55 | +32.24 |
| FNO/residual/hard/next | next | carried | 136.69 | 182.19 | +45.49 |
| FNO/residual/hard/carried | carried | next | 143.71 | 191.49 | +47.78 |
| engine rules/next | next | carried | 114.90 | 155.56 | +40.65 |
| engine rules/carried | carried | next | 113.27 | 156.26 | +42.99 |

1. **Implementing §4.1 literally is what makes the identity an invariant instead of a tolerance.**
   Under the published convention the shells are flagged on
   0.9028-0.9593 of rollout frames once the tolerance is
   0.002 px - the figure 10.53 reported and 10.48 explained - while the carried arms are at
   exactly 0.0000, and the closed-form engine rules move from 0.4103
   to 0.0000 the same way. The price is honest and small: 0.6214
   to 0.6322 test loss, because position stops being
   something the network can fit; and every carried arm's $x$ MAE lands on
   0.1056 px, identical across four unrelated models, which is
   the console's own residue rather than an error any weights are responsible for.
2. **Open-loop drift barely notices the swap, and the closed form is the one place it helps.**
   Paired over five seeds the convention moves drift by
   -1.20
   to +9.51 px
   and not one of those is significant at five seeds. The exception is the model with nothing to
   fit: the engine rules gain -1.64 px
   ($d_z$ = -2.29, $p$ = 0.007) from advancing position with
   $v_t$, which is the same reading 10.49 reached from telemetry alone, now reached through the
   rollout metric.
3. **The convention is not a free re-reading of a trained graph.** Replaying identical weights
   under the other convention costs 25.70-47.78 px of drift on
   every one of the ten arms that can be replayed, always away from the mode it was fitted to. Each
   model learns whichever integration rule it is given, so changing the default is a retraining
   decision, not a config flip - which is exactly why the flags were added rather than the
   defaults moved.
4. **§4.3.5's corrected penalty produces the ground behaviour the recording has, and one family
   pays for it in drift.** The published penalty drives median $|v_y|$ on grounded, un-jumping
   frames to 0.025/1.086/0.150 px/frame
   (MLP/DeepONet/FNO) where the console sits at 6.0; the corrected rule leaves
   2.930/6.437/6.895, with
   $d_z$ = +1.18, +1.32 and
   +8.18 - a third to a full unit of the recorded behaviour. What it costs is
   family-dependent in the direction 10.47 made a habit of:
   +0.76 px (MLP, $d_z$ = +0.39),
   +11.80 px (DeepONet), and +152.83 px
   ($d_z$ = +2.51) for the FNO - while that same FNO's one-step test loss
   *improves* from 5.959 to
   2.473. Truthful about the ground, worse as a
   rollout, for the family that was already the rollout failure of the grid.
5. **The published velocity-violation column reports a bound the graph cannot break.** For the
   eight shell arms and the two closed-form ones the rate at 72.0 is 0.0000 - necessarily,
   because the clamp makes exceeding it unrepresentable - while the same trajectories against the
   reachable 48.0 flag 0.0077-0.1510 of their frames, the top of that range reached by the
   hand-written engine rules. So every "0.0% velocity violation" in Sections 8, 10.27, 10.42, 10.47, 10.53 and
   10.55 is a restatement of the clamp, not evidence that a model stays inside the speeds the
   console has. The arms whose graph bounds nothing are the control on that reading, and they
   behave as the distinction predicts: three of them do exceed 72.0 - the
   `DeepONet/state/soft/published` arm on 0.13% of frames and its corrected twin on 18.80%, and
   `FNO/state/soft/increment` on 14.67% - which is what a penalty that only discourages looks like
   beside a clamp that prevents.
6. **The control passes, so the rest of the paper is untouched.** 30
   arm/seed pairs of the default-convention arms compared against
   `results/operator_physics_injection_metrics.json`: the largest disagreement on drift, $x$ MAE
   and test loss is 0.0 - identical, which is
   what licenses every earlier table to stay as printed while the corrected forms exist beside them.

**What the tables above do not settle.** Five seeds bound every paired statement the way 10.44.1
does, the ground stratum is conditioned on the recorded button latch rather than the engine's
internal jump flag (§10.52's limitation applies verbatim), and the contact rule is tested only on
the `*/soft` cells because the penalty is the sole place it exists as an objective - the shells do
not enforce a ground rule at all, they clamp velocities. What was left unmeasured by this table -
the sign of the correction *in control*, which 10.44/10.47 established that open-loop drift does
not predict - is measured by 10.57.1, and for two of the three families it is the other sign.

#### 10.57.1 The Same Correction on the Real Console

The eight arms that carry a weight - the four shells of the convention table, each trained twice -
are flown where every other controller in this repository is flown: the published CEM-MPC planner,
horizon 15, 256 candidates, three iterations, the same objective weights, the Yoshi's Island 1
savestate, the 300-frame budget and the same five CEM seeds 42-46. The three reference rows run
beside them and reproduce the 10.47 and 10.53 closed loops exactly - `established_wram_engine_rules`
at 619.90 ± 20.02 px, `published_pc_deeponet_10_42` at 635.61 ± 14.18 px and
`published_hard_pinn_10_27` at 299.20 ± 253.32 px, identical worst seed, best seed, frames survived
and death count in all three - so this is the same instrument with eight more controllers in it.

| controller | progress (px) | worst seed | best seed | frames | deaths | vs the established rules |
|:--|--:|--:|--:|--:|--:|:--|
| `established_wram_engine_rules` | 619.90 ± 20.02 | 601.50 | 643.38 | 300.0 | 0 | reference |
| `published_pc_deeponet_10_42` | 635.61 ± 14.18 | 610.25 | 642.31 | 300.0 | 0 | +15.71 px, $d_z$ +0.50, $p$ 0.325 |
| `published_hard_pinn_10_27` | 299.20 ± 253.32 | 113.56 | 579.44 | 224.6 | 3 | -320.70 px, $d_z$ -1.21, $p$ 0.054 |
| `hard_pinn_next` | 434.89 ± 186.21 | 111.25 | 544.06 | 276.2 | 1 | -185.02 px, $d_z$ -0.92, $p$ 0.109 |
| `hard_pinn_carried` | 114.99 ± 1.73 | 112.44 | 116.44 | 173.8 | 5 | -504.91 px, $d_z$ -23.32, $p$ 0.000 |
| `mlp_residual_hard_next` | 434.89 ± 186.21 | 111.25 | 544.06 | 276.2 | 1 | -185.02 px, $d_z$ -0.92, $p$ 0.109 |
| `mlp_residual_hard_carried` | 114.99 ± 1.73 | 112.44 | 116.44 | 173.8 | 5 | -504.91 px, $d_z$ -23.32, $p$ 0.000 |
| `deeponet_residual_hard_next` | 635.61 ± 14.18 | 610.25 | 642.31 | 300.0 | 0 | +15.71 px, $d_z$ +0.50, $p$ 0.325 |
| `deeponet_residual_hard_carried` | 629.69 ± 18.10 | 609.19 | 642.94 | 300.0 | 0 | +9.79 px, $d_z$ +0.35, $p$ 0.473 |
| `fno_residual_hard_next` | 554.74 ± 28.61 | 521.25 | 591.31 | 300.0 | 0 | -65.16 px, $d_z$ -1.47, $p$ 0.030 |
| `fno_residual_hard_carried` | 583.35 ± 26.83 | 544.56 | 602.81 | 300.0 | 0 | -36.55 px, $d_z$ -1.64, $p$ 0.021 |

The four paired contrasts inside the study - same family, same planner, only the velocity that
advances position changing:

* `hard_pinn_next -> hard_pinn_carried`: -319.90 px ($d_z$ -1.73, $p$ 0.018)
* `mlp_residual_hard_next -> mlp_residual_hard_carried`: -319.90 px ($d_z$ -1.73, $p$ 0.018)
* `deeponet_residual_hard_next -> deeponet_residual_hard_carried`: -5.92 px ($d_z$ -0.22, $p$ 0.651)
* `fno_residual_hard_next -> fno_residual_hard_carried`: +28.61 px ($d_z$ +0.67, $p$ 0.208)

1. **The sign of the correction in control is not the sign of its drift, and one family decides it.**
   Open-loop, the same swap moved drift by -1.20 px (the MLP shell arm and the hard PINN arm, which
   train to the same numbers here), +9.51 px (DeepONet) and +7.02 px (FNO), none of them significant
   at five seeds. On the console it costs the MLP/PINN shell -319.90 px ($p$ 0.018), buys the FNO
   +28.61 px ($p$ 0.208, positive in four of the five seeds) and leaves the DeepONet at -5.92 px
   ($p$ 0.651). Which velocity advances $x$ is not bookkeeping inside the integrator: it is what the
   planner believes about the next fifteen frames, and the two readings disagree about the *sign*
   per family.
2. **The arm that satisfies §4.1 exactly is the worst controller in the study.** `hard_pinn_carried`
   is the row whose rollout violation at the 0.002 px ruler is exactly 0.0000 - §4.1 as written, no
   tolerance needed - and whose $x$ MAE is the console's own 0.1056 px. Flown, it dies in 5 seeds out
   of 5 and never passes 116.44 px, while its `next` twin, flagged on 0.9158 of its rollout frames by
   that same ruler, reaches the frame budget in 4 of 5. On the recorded draw it agrees with the
   hand-written rules' program on 0.3410 of its 173 frames against its twin's 0.3467 of 300: two
   controllers that pick the reference action on nearly the same share of frames, one of which never
   leaves the first pit. Exactness of the identity and controllability are different axes, and this
   is 10.44's lesson - open-loop drift does not order controllers - re-enacted on the convention
   instead of the architecture.
3. **A dead controller's mean is a property of the level, not of the driving.** The
   114.99 ± 1.73 px is five deaths between frame 173 and 177, all within 4.00 px of each other, in
   the same hole. Its paired effect size ($d_z$ -23.32) is the largest in the study because the
   differences have almost no spread, not because the gap is better resolved, and its $p$ 0.000 is
   the table's smallest. The published 10.27 PINN falls into that same hole in 3 of the same 5 seeds
   (113.56, 114.06, 115.00 px) and reaches 573.94 and 579.44 px in the other two, which is what
   10.44.1's bimodality of the flagship row is made of. Five seeds support the death count here; the
   -504.91 px against the rules is the pit, not the driving.
4. **Eleven rows, eight programs.** `hard_pinn_next` and `mlp_residual_hard_next` are different
   classes, trained in different runs, loaded from different weight files - and they record identical
   progress in every one of the five seeds, with the identical 300-frame joypad program in the draw
   whose sequence the artifact stores; their `carried` twins do the same over 173 frames.
   `deeponet_residual_hard_next` reproduces the 10.42 published shell's episode to the pixel in all
   five seeds and its recorded sequence too. Grouping the eleven rows by their five-seed progress
   vector, and independently by their recorded action sequence, gives eight programs rather than
   eleven: each of those three pairs is one measurement printed twice, not a replication. It is also
   the strongest evidence this repository has for 10.44's claim that on this level the planner's
   program is insensitive to the difference between two parameterisations of the same structure - it
   is insensitive even to a second training run of one class.
5. **The training run is a rival explanation for the one significant contrast.** `hard_pinn_next`
   (434.89 ± 186.21 px, 1 death) and `published_hard_pinn_10_27` (299.20 ± 253.32 px, 3 deaths) are
   the same class under the same convention and differ by 135.69 px of mean progress - 42% of the
   319.90 px the convention moves. The contrast in finding 1 is paired by CEM seed and this one is
   not, so the design does not put them on the same footing; but what five seeds and a 186 px spread
   can carry for this family is "the corrected arm never gets past the first pit", and not
   "the convention is worth 320 px of driving".

**Limitations of this leg.** Five CEM seeds put the floor of any Wilcoxon test at $p \ge 0.0625$, so
the contrasts are reported with $d_z$ and the per-seed values; one level and one savestate, so the
hole in finding 3 is a fact about Yoshi's Island 1 and not about the console; 6 of the 11 rows reach
the 300-frame budget, so their progress is budget-limited rather than capability-limited; and only
the convention correction is flyable as the study stands - the §4.3.5 ground rule is a term in an
objective, so testing it in control means retraining the `*/soft` cells and flying them (all eight
weights this leg loads are shells), and the velocity-bound correction changes only the evaluator,
which the console never consults, so a planner cannot measure it in principle.

Regenerate: `python -m src.evaluation.physics_injection_mpc_benchmark --study corrected` (requires
the Libretro core and ROM of Section 11.2; 11 controllers x 5 CEM seeds = 55 episodes, 16 min on
this host with the GPU shared with another training job; without hardware it exits with a diagnostic
and writes nothing; writes `results/corrected_physics_mpc_metrics.json`).

**Decision, recorded rather than implied.** The defaults stay as published, and the decision is
now priced on both sides of the same design. Findings 1-3 say what §4.1's literal form costs the
models that were fitted to the other rule, finding 4 the same for §4.3.5, finding 5 that the
number the repository has been quoting as a safety property is a property of a bound its own data
does not reach, and 10.57.1 that the console disagrees with the drift column's sign for two of
the three shells - the corrected MLP/PINN arm dies in the first pit in every seed while the
corrected FNO gains 28.61 px. Regenerate the open-loop half with `python -m
src.evaluation.corrected_physics_ablation`: run without `--save-checkpoints` it writes no weight
at all, so no committed checkpoint is re-dated, and its own `train_seconds` field accounts for
715 s across the 80 fits it logs. `make corrected-physics` passes the flag, which publishes the
eight arms 10.57.1 flies under the study's own `corrphys_` prefix, and then runs the console leg.
