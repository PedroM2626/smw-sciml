[← 10.52](10-52-collecting-the-missing-branch-the-gravity-gate.md) · [Contents](../../README.md#contents) · [10.54 →](10-54-auditing-the-physics-prose-of-section-4.md)

### 10.53 The Velocity the Engine Integrates With: a Prediction Target Nobody Trained

Section 10.49 measured something the repository had never stated consistently: §4.1's identity was $X_{t+1} = X_t + v_{x,t}/16$ - the velocity the frame *starts* with - while the sentence two paragraphs later that defined a "structural kinematic violation" named $\hat v_{x,t+1}$, the velocity the frame *ends* with. §10.54 has since rewritten that sentence with the carried velocity, and records the shells that still integrate with the other one as code disagreeing with the documentation; both readings are implemented in the code, and this section trains them side by side. `DiscreteKinematicsLoss`, `RolloutEvaluator`'s predicate and `rollout_diagnostics` all read the identity with $v_t$; every hard shell since 10.27 advances position with the velocity the network predicts for the next frame. The telemetry arbitrates: over the held-out transitions of this study's own dataset the median of $|\Delta x - v_{x,t}/16|$ is 0.0000 px and the median against $v_{x,t+1}$ is 0.0625 px, one sub-pixel of lag on every accelerating frame, and the console integrates with the carried velocity *exactly* on 93.5% of frames.

So the quantity the engine actually moves by - $v_{\text{eff}} = 16\,\Delta x$, the **effective** velocity - has never been anyone's prediction target here. `src/models/effective_velocity_dynamics.py` makes it one, in three parameterisations that differ in a single line of the graph:

* `next` - $\hat x = x + \hat v_{t+1}/16$: the published shell, rebuilt inside the new class so the convention axis is flipped without touching anything else. It is a *control*, not a re-derivation: across the nine `next` cells and five seeds the multi-start drift reproduces 10.47's `residual` artifact to 0.0 px, so every contrast below is a measured difference between two arms that are identical except for which velocity moves position.
* `carried` - $\hat x = x + v_t/16$: the console's convention. Position stops being a prediction; it is a deterministic function of the input state, and the network has no way to influence it.
* `offset` - $\hat x = x + (v_t + \varepsilon_t)/16$ with $\hat\varepsilon$ a free head: the only form that can represent the frames where the console leaves its own rule. Its two extra channels cost almost nothing - the offset arms have exactly the parameter counts of 10.47's state-output arms (36,744 / 52,744 / 14,537) - so this is the same network capacity, with the deterministic part of the position channel removed.

Three families x three conventions x three mechanisms (none / the published composite penalty / the Section 4 bounds), five seeds, the protocol of 10.42/10.47 unchanged. Unconstrained arms at mechanism `none`:

| family | convention | position error (px) | effective-velocity error (sub-px) | frames integrated exactly | drift (px) | viol @0.2 px | viol @0.002 px | the same test against $\hat v_{t+1}$ @0.002 px |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| MLP | `next` | 0.1359 | 2.17 | 52.9% | 167.1 | 0.0000 | 0.9345 | 0.0000 |
| DeepONet | `next` | 0.1365 | 2.18 | 50.9% | 161.8 | 0.0012 | 0.9578 | 0.0000 |
| FNO | `next` | 0.1312 | 2.10 | 60.2% | 163.4 | 0.0212 | 0.9477 | 0.0000 |
| MLP | `carried` | 0.1056 | 1.69 | 93.5% | 160.4 | 0.0000 | 0.0000 | 0.9331 |
| DeepONet | `carried` | 0.1056 | 1.69 | 93.5% | 156.7 | 0.0000 | 0.0000 | 0.9482 |
| FNO | `carried` | 0.1056 | 1.69 | 93.5% | 145.2 | 0.0000 | 0.0000 | 0.9859 |
| MLP | `offset` | 0.0509 | 0.82 | 63.9% | 146.5 | 0.0200 | 0.9655 | 0.9723 |
| DeepONet | `offset` | 0.2066 | 3.31 | 10.0% | 168.7 | 0.0267 | 0.9872 | 0.9827 |
| FNO | `offset` | 0.0620 | 0.99 | 65.3% | 130.1 | 0.0605 | 0.9615 | 0.9561 |
| console | `carried` | 0.1056 | 1.69 | 93.5% | - | 0.0462 | 0.0462 | 0.8218 |

The same predicate at four tolerances, averaged over families (mean of the `none` cells):

| convention | violation @0.2 px | @0.05 px | @0.01 px | @0.002 px | against $\hat v_{t+1}$ @0.002 px |
|:--|--:|--:|--:|--:|--:|
| `next` | 0.0075 | 0.1131 | 0.7608 | 0.9467 | 0.0000 |
| `carried` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9557 |
| `offset` | 0.0357 | 0.3314 | 0.8401 | 0.9714 | 0.9704 |
| console | 0.0462 | 0.0462 | 0.0462 | 0.0462 | 0.8218 |

All three mechanisms, averaged over families:

| convention | mechanism | position error (px) | drift (px) | kinematic residual the penalty scores (px²) | out-of-bounds |
|:--|:--|--:|--:|--:|--:|
| `next` | `none` | 0.1345 | 164.1 | 0.0014 | 0.188 |
| `next` | `soft` | 0.1282 | 131.1 | 5.0039 | 0.038 |
| `next` | `hard` | 0.1335 | 133.9 | 0.0014 | 0.000 |
| `carried` | `none` | 0.1056 | 154.1 | 0.0000 | 0.188 |
| `carried` | `soft` | 0.1056 | 133.9 | 0.0000 | 0.038 |
| `carried` | `hard` | 0.1056 | 139.0 | 0.0000 | 0.000 |
| `offset` | `none` | 0.1065 | 148.4 | 0.0933 | 0.188 |
| `offset` | `soft` | 0.2119 | 96.4 | 0.0755 | 0.038 |
| `offset` | `hard` | 0.1017 | 139.8 | 0.0945 | 0.000 |

1. **A model that does not predict position at all predicts position better than every model this repository has trained.** The nine `carried` cells share a held-out position error of 0.1056 px with a spread of 0.000000 px across families *and* across mechanisms, because no weight vector touches the position channel in them, and that number is the console's own residual under the same convention (0.1056 px) - it is a measurement of where the engine departs from its rule, not of what a network failed to learn. For comparison, the best position error anywhere in 10.47's 17-arm grid is 0.1184 px (`DeepONet/residual/soft`), so a position head with no parameters in it is 10.8% better than the best learned one, and the `carried` arm is the first in this repository whose position error cannot be improved by training.
2. **The published kinematic-violation figure is a convention test, and this is now measured rather than argued.** 10.48 established that the predicate mixes consistency with smoothness; the ladder above separates the two readings decisively. At the published 0.2 px the `next` arms look perfect (0.0000-0.0212), and so do the `carried` arms - but tighten the tolerance to 0.002 px and the `next` arms are flagged on 0.9467 of their rollout frames while the `carried` arms stay at exactly 0.0000 with no clamp, no penalty and no physics in the model at all. Their zeros are structural; the shells' zeros were a 0.0625 px lag that happens to fit inside 0.2 px. Read the other way - against the model's own next velocity - the picture inverts (0.0000 vs 0.9557), which is the same fact seen from the wrong side of the convention.
3. **The console is the only row that is nonzero on both readings, and a `carried` model is *more* regular than Mario.** The engine integrates with the carried velocity exactly on 93.5% of frames and is flagged by its own predicate on 0.0462 of them - deviations that are all at least 0.2 px, since a frame either obeys the rule or is stopped, wrapped or ridden by a platform. A `carried` network scores 0.0000 there, because it represents the rule and not the exceptions. That is the price the `offset` head pays, and it is visible: giving the network two channels to model the exceptions moves it from 0.0000 to 0.0200-0.0605 - i.e. *toward the console's own 0.0462* - while its position error drops to 0.0509 px (MLP, 51.8% below the `carried` floor) and 0.0620 px (FNO). The DeepONet is the exception in the other direction (0.2066 px, 95.7% worse than `carried`), so the free head is a capability the family has to have, not a free win.
4. **The composite penalty's kinematic term is dead on a carried graph, which re-attributes part of 10.47.** On `carried` cells the trainer's kinematic residual is 0.0000 with and without the penalty - the penalty is satisfied identically, so what 10.47 published as "soft" physics on those graphs can only have been the bound and contact terms. On `next` cells the same term is not dead but *contradicted*: `DeepONet/next/soft` scores 7.3590 px² and `FNO/next/soft` 7.5393, because the graph advances position with $\hat v_{t+1}$ while the penalty demands $v_t$. The penalty is asking for the console's convention and the shell is refusing it; that, and not "physics helping", is at least part of what 10.47's soft cells measured.
5. **On the offset head the penalty is not neutral but destructive, and it destroys exactly the thing the head is for.** Adding $\lambda_{\text{kin}}$ to an `offset` arm suppresses $\hat\varepsilon$ - the mean $|\hat\varepsilon_x|$ on the MLP falls from 1.94 to 0.18 sub-pixels/frame - and the position error goes back to the `carried` floor and past it (0.1065 px mean at `none`, 0.2119 px at `soft`). A residual written against $v_t$ penalises the representation of the frames where the console does not use $v_t$.
6. **Drift and bounds behave as 10.47 said they would, and the convention is orthogonal to both.** The single-axis contrast - same family, same seed, same data term, only the velocity that advances position changed:

| family | metric | `carried` $-$ `next` | effect size | significance |
|:--|:--|--:|:--|:--|
| MLP | multi-start drift (px) | -6.6866 | $d_z$ -1.10 | $p$ 0.070 |
| DeepONet | multi-start drift (px) | -5.1229 | $d_z$ -0.64 | $p$ 0.223 |
| FNO | multi-start drift (px) | -18.1793 | $d_z$ -0.44 | $p$ 0.378 |
| MLP | held-out position error (px) | -0.0302 | $d_z$ -10.26 | $p$ 0.000 |
| DeepONet | held-out position error (px) | -0.0309 | $d_z$ -15.68 | $p$ 0.000 |
| FNO | held-out position error (px) | -0.0255 | $d_z$ -7.83 | $p$ 0.000 |
| MLP | effective-velocity error (sub-px) | -0.4837 | $d_z$ -10.26 | $p$ 0.000 |
| DeepONet | effective-velocity error (sub-px) | -0.4943 | $d_z$ -15.68 | $p$ 0.000 |
| FNO | effective-velocity error (sub-px) | -0.4084 | $d_z$ -7.83 | $p$ 0.000 |
| MLP | violation at 0.002 px | -0.9345 | $d_z$ -21.01 | $p$ 0.000 |
| DeepONet | violation at 0.002 px | -0.9578 | $d_z$ -65.30 | $p$ 0.000 |
| FNO | violation at 0.002 px | -0.9477 | $d_z$ -16.04 | $p$ 0.000 |

   The convention is consistent in sign on drift in all three families but individually insignificant at five seeds; it is decisive on the position and predicate cells, where the paired difference is a constant because one side of it has no learnable position channel at all. The clamp still does its own job in either convention: out-of-bounds 0.188 under `none` and 0.000 under `hard` in every convention, with the convention contributing nothing to boundedness.
7. **Closed loop, the convention is the difference between a controller and none.** Same planner, objective, savestate, 300-frame budget and five CEM seeds as 10.44/10.47, and the published `next` arm flown beside each `carried` and `offset` arm so the convention is the only difference between rows:

| controller | progress (px) | worst seed | best seed | frames | deaths | vs the established rules |
|:--|--:|--:|--:|--:|--:|:--|
| `established_wram_engine_rules` | 619.90 ± 20.02 | 601.50 | 643.38 | 300.0 | 0 | reference |
| `published_pc_deeponet_10_42` | 635.61 ± 14.18 | 610.25 | 642.31 | 300.0 | 0 | +15.71 px, $d_z$ +0.50, $p$ 0.325 |
| `published_hard_pinn_10_27` | 299.20 ± 253.32 | 113.56 | 579.44 | 224.6 | 3 | -320.70 px, $d_z$ -1.21, $p$ 0.054 |
| `mlp_next_none` | 498.84 ± 216.13 | 114.44 | 634.06 | 274.6 | 1 | -121.06 px, $d_z$ -0.53, $p$ 0.305 |
| `mlp_carried_none` | 521.01 ± 228.70 | 113.12 | 637.00 | 274.8 | 1 | -98.89 px, $d_z$ -0.41, $p$ 0.412 |
| `mlp_offset_none` | 622.95 ± 18.32 | 602.94 | 643.06 | 300.0 | 0 | +3.05 px, $d_z$ +0.14, $p$ 0.772 |
| `deeponet_next_none` | 623.06 ± 17.55 | 610.12 | 643.00 | 300.0 | 0 | +3.16 px, $d_z$ +0.09, $p$ 0.842 |
| `deeponet_carried_none` | 637.46 ± 11.62 | 616.69 | 643.06 | 300.0 | 0 | +17.56 px, $d_z$ +0.60, $p$ 0.249 |
| `deeponet_offset_none` | 633.98 ± 18.20 | 601.44 | 642.94 | 300.0 | 0 | +14.07 px, $d_z$ +0.75, $p$ 0.171 |
| `fno_next_none` | -4.25 ± 2.98 | -7.50 | -0.88 | 300.0 | 0 | -624.15 px, $d_z$ -35.66, $p$ 0.000 |
| `fno_carried_none` | 468.69 ± 212.45 | 98.44 | 610.38 | 281.0 | 1 | -151.21 px, $d_z$ -0.74, $p$ 0.174 |
| `fno_offset_none` | 603.38 ± 3.15 | 600.88 | 606.94 | 300.0 | 0 | -16.52 px, $d_z$ -0.84, $p$ 0.134 |

   The FNO row is the result. With the published convention it makes no progress at all over five seeds ($-4.25 \pm 2.98$ px, i.e. it ends up *behind* the start), and the identical network trained under the console's convention reaches 468.69 px; the `offset` head takes the same family to 603.38 px with a between-seed spread of 3.15 px against the hand-written rules' own 20.02 px - the unconstrained FNO stops being a lottery. Every paired contrast inside the study, same family and same planner, only the convention changing:

* `mlp_next_none -> mlp_carried_none`: +22.17 px ($d_z$ +0.52, $p$ 0.311)
* `mlp_next_none -> mlp_offset_none`: +124.11 px ($d_z$ +0.55, $p$ 0.286)
* `mlp_carried_none -> mlp_offset_none`: +101.94 px ($d_z$ +0.42, $p$ 0.397)
* `deeponet_next_none -> deeponet_carried_none`: +14.40 px ($d_z$ +0.87, $p$ 0.123)
* `deeponet_next_none -> deeponet_offset_none`: +10.91 px ($d_z$ +0.56, $p$ 0.278)
* `deeponet_carried_none -> deeponet_offset_none`: -3.49 px ($d_z$ -0.15, $p$ 0.760)
* `fno_next_none -> fno_carried_none`: +472.94 px ($d_z$ +2.24, $p$ 0.007)
* `fno_next_none -> fno_offset_none`: +607.63 px ($d_z$ +227.20, $p$ 0.000)
* `fno_carried_none -> fno_offset_none`: +134.69 px ($d_z$ +0.64, $p$ 0.226)

   Six of the nine contrasts move away from the published convention and all six are positive, from +10.91 px to +607.63 px; the three that compare the two new conventions against each other are +101.94, +134.69 and -3.49 px, the last of them the only row in the study where the free head costs progress. Every `offset` arm survives all five seeds to the frame budget with zero deaths, while `mlp_next_none` and `mlp_carried_none` each die once, `fno_carried_none` dies once, and the published `Hard_PINN` of 10.27 dies three times.
   The control side of the parity check holds too: `mlp_next_none` reproduces 10.47's `mlp_residual_none` row exactly (498.84 ± 216.13, min 114.44, max 634.06), `deeponet_next_none` reproduces `deeponet_residual_none` (623.06 ± 17.55) and `fno_next_none` reproduces `fno_residual_none` (-4.25 ± 2.98), so the gains above are the same weights under one changed line of the graph rather than a second training run.

**Limitations.** The `carried` arm's position error is not a model quality but a property of the recording, and it is stage-specific: 0.1056 px on the horizontal channel of the gameplay recording, 0.6253 px vertically, where the exceptions are far more common (73.9% of frames exact against 93.5%) because platforms move and the ceiling and floor intervene - so the free `offset` head has much more to learn vertically, and the vertical gains are where its risk lies. Five seeds put a floor of $p \ge 0.0625$ on any paired test here, so the closed-loop effects are reported with $d_z$ and the raw per-seed range, and the drift gains in finding 6 are individually insignificant. The rollout ladder is computed on 10-start, 120-frame rollouts of the held-out split, which is where the exceptions concentrate; a policy-driven rollout would meet wall stops and screen wraps in a different mix. `offset` costs two extra output channels per arm, and the DeepONet's failure with them (0.2066 px) is unexplained beyond "this family did not use the head". Finally, 10.44's lesson still holds and is visible in these very rows: open-loop drift ordered the FNO arm last and the closed loop orders it first, so the convention's control gain is not predicted by its 6.7 px drift gain.

![Which velocity advances position](../../results/figures/effective_velocity.png)

*Left: held-out position error per unconstrained arm on a log scale, with the console's own residual as the dashed line - the three `carried` bars sit exactly on it. Right: the published violation predicate against the tolerance ladder, with the console's rate dashed: the `carried` curves do not rise because there is no tolerance at which a structural identity fails.*

Regenerate: `python -m src.evaluation.effective_velocity_benchmark --seeds 42,43,44,45,46` (emulator-free, ~14 min on the RTX 4070 for 27 arms x 5 seeds; `--arms MLP` cuts it to a third, smoke config `configs/smoke_effective_velocity.yaml`), then `python -m src.evaluation.physics_injection_mpc_benchmark --study effective` (hardware, ~12 min, writes `results/effective_velocity_mpc_metrics.json`), or `make effective-velocity` for both.
