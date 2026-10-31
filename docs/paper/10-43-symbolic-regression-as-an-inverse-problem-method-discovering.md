[← 10.42](10-42-physics-constrained-deeponet-and-the-fourier-neural-operator.md) · [Contents](../../README.md#contents) · [10.44 →](10-44-closed-loop-control-with-inverse-problem-world-models.md)

### 10.43 Symbolic Regression as an Inverse-Problem Method: Discovering the Law

Section 10.40 solves the inverse problem *parametrically*: the traction / friction / asymmetric-gravity map of Section 4 is posited in full, only its seven constants are unknown, and Adam plus a Laplace posterior recovers them. That answer is bounded by the quality of the posited structure - if the assumed law is wrong, the posterior is a precise statement about the wrong model. This section removes the structure from the assumptions and asks the harder question: **can the engine's law be discovered from transitions alone, with nothing posited?**

The tool is tree genetic programming (`gplearn` 0.4.2; subtree crossover and mutation over the piecewise-affine primitive set $\{$`add`, `sub`, `mul`, `div`, `neg`, `abs`, `min`, `max`$\}$, with gplearn's protected division), applied to four one-step laws rather than to the next state directly:

$$\Delta v_x = f_1(v_x, d, \rho),\qquad \Delta v_y = f_2(v_y, \jmath, c_{\text{ground}}),\qquad \Delta x = f_3(v_x^{t+1}),\qquad \Delta y = f_4(v_y^{t+1})$$

Velocity channels are fitted as *increments*, because their physical magnitude *is* an acceleration (~1 sub-pixel/frame), which keeps the constants the search must evolve at $O(1)$; position channels are fitted against the **post-step** velocity, the form in which the discrete integration identity $\hat X_{t+1} = X_t + \hat v_x/\sigma$ becomes a univariate fit that either is or is not discovered. Composing the four laws reproduces `simulate_step`'s bookkeeping exactly (`symbolic_step` / `model_rollout` in `src/inverse/symbolic_regression.py`), so the discovered model and the parametric model of 10.40 are rolled out, probed and scored by identical code - and `tests/test_symbolic_regression.py` asserts that the analytic constants driven through that same interface reproduce `simulate_rollout` to float tolerance.

**Excitation normalisation (not optional).** GP terminals are drawn uniformly from a bounded `const_range` ($[-4,4]$ here). The horizontal law mixes a per-frame increment of ~1 sub-pixel with a rigid ceiling of 48 sub-pixels - two constants a factor of 48 apart, so *no* single terminal range expresses both. Every law is therefore fitted with each feature and its target divided by the maximum its own **fit rows** reach (the evaluation rows never touch the scale), which puts the accelerations, the friction decrement and the integration slope inside the searchable range. The consequence is measured, not argued: the ceiling can then only appear as a *fixed point* of the discovered map, so the probe stage searches for one and reports "no fixed point" rather than substituting the data maximum.

**The probe stage: from expression to physics.** A symbolic law is not a parameter vector, and GP's evolved terminal values are notoriously poor estimates of one. Every constant is therefore defined as a *response* of the discovered map at designed probes: $\tau_{\text{walk}}$ / $\tau_{\text{run}}$ are the median driven increment over the lower half of the observed speed range with and without the run button; µ the median coasting decrement; $g_{\text{hold}}$ / $g_{\text{fall}}$ the median increment on ascending probes with and without jump held, plus a descending consistency probe; $\sigma$ the reciprocal slope of the $f_3$ fit, reported with its worst residual from that straight line; and $\max_{v_x}$ the fixed point of the driven map - accepted **only if the map actually accelerates below it**, because a degenerate zero-drive law satisfies $\hat{v} \le v$ everywhere and would otherwise be misread as a bound at the bottom of the probe range.

**Protocol.** 6 independent replicates (a fresh hidden-world data draw per replicate, never a resplit of one bank), 3 GP seeds per law, population 500, 25 generations, $\leq 4000$ fit transitions per law, parsimony coefficient $10^{-3}$; the parametric comparison is refit by Adam (900 steps) on *exactly* the same thinned windows, so the pairing controls the data and varies only the estimator. Symbolic vs parametric is tested per replicate with paired Wilcoxon, paired $t$ and Cohen's $d_z$ (the 10.39.1 protocol). Emulator-free, CPU-only, deterministic: 270 GP fits (72 in S1, 18 in the budget control, 108 in the reweighting control, 36 in S2, 24 in S3, 12 in S4), ~20 min wall-clock, artifact `results/symbolic_inverse_metrics.json`, figure `results/figures/symbolic_inverse_recovery.png`.

#### 10.43.1 The Discovered Expressions

The three seeds of replicate 0, verbatim (`dir` $\equiv$ signed direction, `run` $\equiv \rho$, `jump` $\equiv \jmath$, `gnd` $\equiv$ ground contact):

```
dvx[0]  min(max(0.491, run), sub(dir, div(vx, 2.289)))
dvx[1]  mul(max(run, 0.590), dir)
dvx[2]  dir
dvy[0]  div(div(div(vy, neg(ground)), 2.322), 2.322)
dvy[1]  div(abs(vy), div(3.850, div(0.197, div(abs(sub(vy, jump)), 3.649))))
dvy[2]  abs(div(max(-0.355, vy), -1.654))
dx[*]   vx_next            (identical in all 3 seeds and all 6 replicates)
dy[*]   vy_next            (identical in all 3 seeds and all 6 replicates)
```

The horizontal law is found in a genuinely readable form: `mul(max(run, 0.590), dir)` *is* $\Delta v_x = d\,(\tau_{\text{walk}} + \rho\,(\tau_{\text{run}} - \tau_{\text{walk}}))$, with the traction tiers probed at 1.062 and 1.800 sub-pixel/frame against the hidden world's 1.000 and 1.800. The vertical law is where the search flounders: two of the three expressions above never reference `jump` at all, and the third buries it inside a ratio tower.

Mean over the 6 replicates (min/max columns are means of the per-replicate extremes; "null R²" is the fit-mean predictor on the same held-out rollouts):

| Law | Held-out MAE (sub-px or px) | Held-out R² (mean / best seed) | Null R² | Beats null | Nodes (mean / min / max) | Seconds per fit | Clamp used | Input gate used |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\Delta v_x$ traction + friction | 0.357 ± 0.069 | 0.820 / 0.875 | $-9\times 10^{-4}$ | 100% | 6.8 / 3 / 11 | 4.0 | 83% | 100% |
| $\Delta v_y$ gravity + floor | 0.897 ± 0.405 | 0.249 / 0.446 | $-3\times 10^{-3}$ | 100% | 7.7 / 5 / 11 | 4.3 | 28% | 22% |
| $\Delta x$ integration | $1.9\times 10^{-5}$ | **1.0000** / 1.0000 | $-1\times 10^{-3}$ | 100% | **1.0 / 1 / 1** | 3.8 | 0% | - |
| $\Delta y$ integration | 0.00736 | 0.9865 / 0.9865 | $-3\times 10^{-3}$ | 100% | **1.0 / 1 / 1** | 3.7 | 0% | - |

"Input gate used" is the fraction of GP draws whose expression references the discrete switch at all (`dir`/`run` horizontally, `jump` vertically). On the synthetic world every law beats the mean predictor, and the horizontal input dependence is recovered in 100% of draws - while **the held-jump gravity gate is absent from 78% of the discovered vertical laws**. That asymmetry is the seed of the whole result: one branch is an affine response the fitness rewards immediately, the other is a discontinuous conditional whose payoff is small in the error metric and expensive in the search.

#### 10.43.2 Constant Recovery: Structural Search vs Posited Structure (S1)

Median relative error over the 6 replicates (per replicate: bagged over the 3 GP seeds), against the Section 10.40 estimator refit on the same windows:

| Constant | Hidden world | Prior (SMW) | **Symbolic GP + probe** | Single-seed GP | Parametric ID (10.40) | GP recovery rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\max_{v_x}$ velocity ceiling | 48.0 | 50.0% | **no fixed point (0/6)** | 0/6 | 0.0006% | 0% |
| $\tau_{\text{walk}}$ | 1.000 | 25.0% | 22.2% ± 16.0 | 60.5% | 0.003% | 100% |
| $\tau_{\text{run}}$ | 1.800 | 16.7% | **3.6%** | 10.8% | 0.002% | 100% |
| $\sigma$ sub-pixels per pixel | 20.0 | 20.0% | **0.001%** | 0.001% | 0.589% | 100% |
| $g_{\text{hold}}$ | 2.400 | 25.0% | 78.8% ± 19.3 | 96.7% | 2.385% | 100% |
| $g_{\text{fall}}$ | 5.200 | 15.4% | **3.5%** ± 4.8 | 9.2% | 0.231% | 100% |
| µ coast friction | 0.600 | 16.7% | 89.2% (probed 0.065) | 67.5% | 0.004% | 100% |

Aggregate accuracy: bagged symbolic $1.15\times 10^{-2}$ vs parametric $8.12\times 10^{-5}$ on the channel-variance-weighted one-step metric, over 6 paired replicates - **paired Wilcoxon $p = 0.03125$ (the exact-sample floor at $n = 6$), paired $t$-test $p = 5.6\times 10^{-4}$, Cohen's $d_z = 3.18$**; selecting the best GP seed per replicate instead of bagging gives $9.52\times 10^{-3}$, $d_z = 2.33$. The same pairing against the *kinematic-persistence* null (no force law, exact integration) gives $1.15\times 10^{-2}$ vs $2.08\times 10^{-2}$, Wilcoxon $p = 0.03125$, $d_z = -4.97$: the discovered dynamics is a real, significant improvement over doing nothing, and still two to three orders of magnitude behind the estimator that was handed the law.

Three conclusions, each of which is a *discovery* result rather than a failure:

1. **The discrete integration identity is rediscovered exactly, in every replicate and every seed.** $R^2 = 1.0000$, a 1-node program, probed $\sigma = 19.9998$ (0.001% from the hidden world's $\sigma = 20$) with a worst-case non-linearity of $4.4\times 10^{-16}$ px, and the independent $y$-channel probe agrees: 19.999996. Here the symbolic estimator is *better* than the parametric one (0.589% error - the published 10.40-E1 behaviour, where $\sigma$ is partly identify-then-compensate against the velocity channels over a multi-step rollout) because $f_3$ is a univariate regression with nothing to trade off. The kinematics of Section 4 is therefore the one ingredient that needs no physics prior; the structure-free baselines of Section 8 fail at it not because the identity is unknown to them but because they never regress the right variable.
2. **The rigid bound is not expressible by the searched representation.** 83% of $\Delta v_x$ programs use `min`/`max`, yet none of the 18 single-seed horizontal laws of S1, nor any of the 9 in the budget sweep, nor any of the 6 bagged replicates produces a map with a fixed point inside the explored range, and the rolled-out model overshoots the true ceiling by 1.54 sub-pixel/frame on average (up to 49.5 on the worst single draw, S2). A clamp at ±48 needs a terminal of magnitude 48 under a $[-4,4]$ `const_range` *and* must be applied at the right place in the tree, while the mean-error fitness and the parsimony term give no reason to try. This is the structural-level analogue of Section 10.40's inactive-constraint pathology: there, gradient descent could not lower a ceiling its rollout never reached; here, the search cannot represent a ceiling its terminals cannot reach. The remedies differ, and that is the point - 10.40 fixes it with a warm start from the observed velocity range, whereas the symbolic estimator needs the one-sided constraint *positured as a primitive*, which is exactly what Section 5.4's hard-residual shell does.
3. **The discontinuous gravity gate is the hardest term to discover.** $g_{\text{fall}}$ comes out at 3.5% error, $g_{\text{hold}}$ at 78.8% (single-seed 96.7%), and the bagged vertical law separates the two tiers by 0.00 / -1.17 / 0.00 sub-pixel/frame at the small / medium / large budgets against a true separation of -2.80. With 78% of programs never referencing `jump`, the correct reading is that the observables are insufficient *for the fitness*, not that the optimiser underperformed: fitting the 5.2 sub-pixel falling branch everywhere already captures most of the squared error. The descending consistency probe (5.56 ± 0.60 vs the ascending 5.05 ± 0.37) shows the same single-tier collapse from the other side.

#### 10.43.3 Two Controls: Search Effort and Experimental Design

"Representation cannot express it" and "the search was unlucky" look identical in a single accuracy number, so S1 runs two controls.

**S1b - budget sweep.** Population × generations over a 7.5× range, reporting held-out R² *and* the two structural read-outs:

| Budget (pop × gens) | $\Delta v_x$ R² | $\Delta v_x$ nodes | fixed-point rate | $\Delta v_y$ R² | gravity-tier separation (true -2.80) | seconds/fit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $300 \times 15$ | 0.802 ± 0.077 | 9.0 | **0.00** | 0.133 ± 0.181 | 0.00 | 1.5 |
| $500 \times 25$ | 0.809 ± 0.080 | 5.0 | **0.00** | 0.379 ± 0.250 | -1.17 | 4.0 |
| $1000 \times 60$ | 0.873 ± 0.056 | 23.3 | **0.00** | 0.624 ± 0.459 | 0.00 | 22.6 |

Accuracy climbs monotonically (the vertical law's R² goes $0.13 \to 0.62$), so the search is not starved. Structure does not follow: the fixed-point rate stays at 0 of 18 draws across a 15× budget increase, and the tier separation moves $0.00 \to -1.17 \to 0.00$ with no trend and a seed scatter (±0.46) larger than its mean. This is the cleanest available separation of the two hypotheses - the ceiling is a *representation* limit and the gate a *fitness-landscape* limit, and neither is solved by more compute.

**S1c - excitation-reweighted fitting.** Section 10.40's identifiability lesson is that a constant is recoverable only if the data excite the term that uses it. Weighting coast frames (where friction acts) and simultaneously ascending + jump-held frames (where $g_{\text{hold}}$ acts) inside the GP fitness tests the same claim for *functional form*:

| Weight strength | coast-friction error | $g_{\text{hold}}$ error | $\Delta v_x$ hold MAE | $\Delta v_y$ hold MAE |
| :---: | :---: | :---: | :---: | :---: |
| 0 (passive observation) | 89.2% | 81.3% | 0.362 | 0.968 |
| 3 | 86.5% | 45.3% | 0.366 | 0.726 |
| 10 | **59.0%** | **11.6%** | 0.453 | 1.539 |

Experiment design buys structure: an 8× cut in $g_{\text{hold}}$ error and a 1.5× cut in friction error, paid for with a 25-59% degradation in aggregate held-out accuracy. That is the optimal-experiment-design trade-off in its pure form, and it extends Section 10.40's identifiability argument from *constants* to *functional form* - with the active-excitation programme of Section 10.12 as the RL-side counterpart.

#### 10.43.4 Long-Horizon Behaviour of a Discovered Model (S2)

120-frame open-loop rollouts in the hidden world (3 replicates × 40 starts), scored against the *true* ceiling and the *true* fixed-point scale:

| World model | Mean drift (px) | Final drift (px) | Frames above the velocity cap | Max cap overshoot (sub-px/frame) | Integration residual (px) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Oracle true world | 0.45 ± 0.14 | 0.54 | 0.0% | 0.000 | $2\times 10^{-14}$ |
| Parametric ID (10.40) | 0.93 ± 0.15 | 1.69 | **0.0%** | 0.000 | 0.0130 |
| Prior SMW constants | 63.63 ± 3.70 | 166.82 | 74.0% | 24.000 | 0.7318 |
| **Symbolic GP (single seed)** | **24.20 ± 1.33** | **64.98** | **68.7%** | 49.501 | $2.8\times 10^{-5}$ |
| **Symbolic GP (bagged, 3 seeds)** | 101848.84 | 2461555.52 | 83.4% | 93.849 | $3.9\times 10^{-5}$ |

Two things read cleanly off this table. First, the discovered model's *integration* is essentially exact ($2.8\times 10^{-5}$ px residual, versus 0.0130 px for the parametric model, whose $\sigma$ is slightly mis-identified): the kinematic identity is not the problem. Second, its *reachable set* is unbounded - 68.7% of rollout frames sit above the engine's real velocity ceiling, with a worst-case overshoot of 49.5 sub-pixel/frame. And bagging, which helps accuracy everywhere else, makes this dramatically worse: the mean of three non-saturating laws is still non-saturating, its per-step drive sits nearer the linear-regime slope, and the 120-frame drift blows up by four orders of magnitude. **Averaging models that all violate a constraint does not restore the constraint.** Paired across the 3 replicates, symbolic drift exceeds parametric drift with Cohen's $d_z = 0.586$ but only Wilcoxon $p = 0.25$ (the heavy tail dominates the mean and $n = 3$ caps the rank statistic); on cap violations the separation is unambiguous - $d_z = 65.4$, paired $t$-test $p = 7.8\times 10^{-5}$ - because the parametric estimator's violations are exactly 0 in every replicate. The honest summary is that the drift comparison is *not* statistically resolved at this replicate count, while the constraint comparison is.

This is Section 8's central thesis reproduced for a *discovered* model instead of a learned one: predictive accuracy and constraint adherence are separate axes, ensembling is not a substitute for structure, and a world model that cannot represent a rigid bound will eventually walk through it.

#### 10.43.5 Genuine WRAM Telemetry (S3)

The same four laws, re-discovered on the canonical training split (2,110 fit transitions after a stride of 3, 1,356 test transitions; the "real" feature preset adds all four terrain-contact channels) and scored per channel on the canonical test split with the Section 7 protocol's metric. The contact byte is supplied to both inverse models as an exogenous input, exactly as in 10.40-E2; the learned baselines must instead infer it, which is why their $x$ and $y$ numbers are not directly comparable to the two rows above it.

| Model | Paradigm | Test MSE $x$ (px²) | Test MSE $y$ | Test MSE $v_x$ | Test MSE $v_y$ | Weighted MSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Symbolic GP (bagged, Ours)** | Discovered law, structure-free | 0.1367 | **0.4603** | 3.727 | **78.48** | **0.01383** |
| Kinematic persistence (null: no forces, exact $\sigma$) | Floor for the comparison | **0.1348** | 3.156 | 3.727 | 78.48 | 0.01440 |
| Parametric ID (10.40-E2) | Posited structure, fitted constants | 0.1723 | 2.665 | **2.345** | 523.69 | 0.07462 |
| Analytic prior (WRAM constants) | Posited structure, no fit | 0.1401 | 3.139 | 2.616 | 523.81 | 0.07495 |
| Hard Residual PINN (10.27, published) | Hard inductive bias, learned | 0.1341 | 1.779 | 3.419 | 138.30 | - |
| Statistical MLP (10.27, published) | Black-box learned | 18225.28 | 3622.22 | 9.04 | 104.21 | - |
| DeepONet (10.41, published) | Learned operator | 274.02 | 744.27 | 140.44 | 1772.23 | - |
| FNO (10.42, published) | Learned spectral operator | 0.63 | 1.62 | 4.03 | 77.73 | - |

The null row is what makes this table readable rather than triumphant. On real telemetry the discovered *velocity* laws do not beat the mean predictor (R² of $-1.7\times 10^{-4}$ against a null of $-1.4\times 10^{-4}$ for $\Delta v_x$; only 33% of seeds beat it at all) and they degenerate to bare contact terminals - `left`, `ceiling` - i.e. the search found the collision byte, not the traction or gravity law. The discovered *position* laws do beat it decisively ($\Delta x$: $R^2 = 0.890$ against a null of -0.157, again the 1-node identity; $\Delta y$: 0.868 against -0.0004, this time a 9.3-node clamping law `max(sub(-1.043, vy_next), vy_next)` that encodes the floor). Consequences, all four of which the artifact records:

1. **GP's composed model beats the posited-but-misspecified analytic map on 3 of 4 channels** ($x$ 0.137 vs 0.172, $y$ 0.460 vs 2.665, $v_y$ 78.5 vs 523.7) and by 5.4× on the weighted metric - but the null row shows most of that is the identity plus the contact byte, not discovered dynamics: on $v_x$ and $v_y$ the two rows are *identical* because the discovered velocity law is a zero increment. The correct claim is the uncomfortable one: **on real gameplay, "no force law at all, exact integration, given collision flags" out-predicts the 10.40-E2 analytic map**, whose own constants sit up to 84% from the reverse-engineered values (10.40-E2: $\tau_{\text{walk}}$ 84.1%, $\tau_{\text{run}}$ 65.7%, $g_{\text{hold}}$ 55.8%, $g_{\text{fall}}$ 65.9%, $\sigma$ 33.8% - 10.40's own honest conclusion). Model-form error costs more than the absence of dynamics.
2. **The one genuinely new physical constant GP recovers on real data is the fixed-point scale**: probing the discovered $\Delta x$ law gives $\sigma = 15.89$ sub-pixels/pixel against the reverse-engineered 16.00 - a **0.69%** error, obtained from gameplay telemetry with the WRAM value never told to the estimator, and *better* than 10.40-E2's multi-step parametric fit of the same constant (33.8% from the reference, because the rollout fit trades $\sigma$ against the mis-identified accelerations).
3. **The $\Delta y$ floor clamp is a real discovery**: `max(-1.043 - v_y^{t+1},\; v_y^{t+1})` is a one-sided reset of downward velocity expressed in the *post-step* velocity, and it is the reason GP's $y$ error (0.460) is below both the analytic prior (3.139) and the published Hard PINN (1.779) on that channel - the learned model must infer ground contact from the state, GP was handed the contact byte and symbolically compressed it into a clamp.
4. **Nothing here contradicts Section 8's ordering.** On the channels that require learning a force law from data, the published Hard PINN remains the reference; the symbolic row is competitive only where the increment is either exactly linear (position) or nearly zero (velocity, given the contact flags).

The grey-box control (`hybrid_residual_control`) asks the sharpest version of the same question: fit GP to the *residuals of the identified analytic model*, offering it all eleven observable channels. If the analytic model's remaining error had a closed form in the observation, this is where it would appear.

| Residual channel | Train R² | Test R² | Best test R² | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| $x$ | +0.101 | +0.228 | +0.256 | partly closed-form |
| $y$ | +0.388 | +0.449 | +0.734 | partly closed-form |
| $v_x$ | -0.018 | -0.003 | -0.003 | **not closed-form** |
| $v_y$ | +0.336 | +0.409 | +0.702 | partly closed-form |

Vertical position, vertical velocity and horizontal position residuals are partially predictable out-of-sample ($R^2 \gt 0.2$, up to 0.73 for the best seed on $y$) - these are the collision-response and floor-contact effects the analytic model approximates crudely. The **horizontal velocity residual is not**: its train R² is already negative, so the identified model's horizontal error is not a learnable function of state, buttons and contact flags at all. The reading offered in this section's first draft was that the missing variable is tile geometry - and Section 10.43.8 now tests exactly that against the repository's own terrain recording, which refutes it for $v_x$ and confirms it for $v_y$. What survives is the estimator-side statement, which is what the experiment actually establishes: symbolic search separates the part of a model's form error that is a closed form of the observation from the part that is not, and this is the third time in this repository (after Section 10.20's $\eta = 10^{-6}$ degeneracy and Section 10.34's tilemap analysis) that the honest ceiling on real-data accuracy turns out to be what is measured rather than how it is fitted.

#### 10.43.6 Zero-Shot Control Transfer of a Discovered Law (S4)

The Section 10.40-E3 held-out battery (400 control sequences executed in the hidden world; signed predicted-minus-achieved final $x$), with the symbolic models added and the 10.40 rows recomputed as a comparability check:

| World model | Transfer MAE (px) | Relative MAE | Optimism (px) | 10.40 published MAE |
| :--- | :---: | :---: | :---: | :---: |
| Oracle true world | $6\times 10^{-5}$ | 0.0% | $+2\times 10^{-6}$ | 0.00 (matches to float32 noise) |
| Parametric ID (this study, 900 steps) | 0.32 | 0.09% | -0.01 | 0.28 (2000 steps) |
| **Symbolic GP (single seed)** | **2.15** | **0.62%** | -0.62 | - |
| **Symbolic GP (bagged, 3 seeds)** | **3.47** | **1.00%** | -0.15 | - |
| Prior SMW constants | 11.17 | 3.24% | +0.26 | 11.17 (matches to $8\times 10^{-6}$ px) |

The prior remains the only systematically *optimistic* model (+0.26 px), exactly as 10.40 reported, and its transfer error is 3.2-5.2× the symbolic model's. The symbolic rows are *pessimistic* (-0.62, -0.15 px): a non-saturating law under-predicts how quickly Mario is pinned against the ceiling, so the planner it feeds is conservative. A discovered model therefore inherits the opposite bias from the one Section 10.40 measured - in a hazard-evasion task (Section 10.8) the safe direction, but a bias nonetheless, and one that bagging does not remove: averaging cuts optimism by 4× yet *raises* transfer MAE from 2.15 to 3.47 px.

#### 10.43.7 What the Symbolic Inverse Establishes

1. **The kinematics is discoverable; the constraints are not - by this estimator.** The discrete integration identity is recovered exactly by every GP draw of every replicate (1 node, $R^2 = 1.0000$, $\sigma$ to 0.001% on synthetic and 0.69% on real telemetry), while the rigid velocity bound yields zero fixed-point discoveries in 18 draws across a 15× budget range. What Section 5.4's hard shell contributes is precisely the part that data alone does not supply - and Section 10.43.9 narrows that sentence to what it can support: the same windows *do* identify the ceiling (a clamp offered as a candidate is recovered at 48.000 and preferred by every selection criterion), and a tree search given three times the budget discovers it unaided - so what data alone does not supply is neither the evidence nor, in the end, the expressible form; it is the search budget.
2. **Knowing the structure buys two to three orders of magnitude.** On identical data the posited-structure estimator beats the discovered-structure estimator on the weighted one-step metric ($8.1\times 10^{-5}$ vs $1.15\times 10^{-2}$; Wilcoxon $p = 0.03125$, $d_z = 3.18$) and on six of the seven constants. The Bayesian machinery of 10.40 (Laplace covariance, Fisher spectrum, Metropolis cross-check) additionally has no symbolic analogue at this budget: a tree is not a differentiable parameterisation, so "how uncertain is this discovered law?" cannot be answered with $J^{\top}J$. The seed-to-seed spread of the probed constants ($\pm 19.3\%$ on $g_{\text{hold}}$, $\pm 16.0\%$ on $\tau_{\text{walk}}$) is the honest substitute, and it is wide.
3. **Search effort buys accuracy, not structure.** R² rises from 0.13 to 0.62 on the vertical law while its tier separation stays statistically at zero. Anyone citing "symbolic regression discovers the laws of a system" should cite the accuracy and structure columns separately, as done here.
4. **Excitation is a binding constraint on discovery, not only on identifiability.** Over-weighting coast and ascending-jump-held frames cuts $g_{\text{hold}}$ error from 81.3% to 11.6% and friction error from 89.2% to 59.0%, at a measurable cost in aggregate accuracy.
5. **A null baseline is not optional in equation discovery.** Every per-law row in this section is reported against the fit-mean predictor, and the composed model against kinematic persistence. Without them the real-data table would appear to show symbolic regression beating the Hard Residual PINN; with them it shows the far less exciting and far more useful truth - that on real gameplay most of the accuracy is the identity plus the contact byte, and that the discovered *dynamics* is where the method stops.
6. **On real telemetry the limit is what is measured, not how it is fitted - and the specific missing variable is testable.** The horizontal-velocity residual of the identified analytic model has negative train R² even with all eleven observable channels offered to the search; Section 10.43.8 then hands the same search the terrain and finds that occupancy explains the *vertical* residual (gain +0.054 against a placebo at -0.007) and not the horizontal one (gain exactly +0.000). Symbolic regression earns its place here as the diagnostic that makes both statements checkable - and, in the $y$ channel, as the design that catches a +0.077 apparent gain as overfitting.

**Limitations.** (i) One GP implementation and one primitive set: transcendental primitives, an ephemeral-constant range matched to the ceiling, or a template admitting a clamp as a first-class node would change conclusions (1) and (2) - what is measured here is what *this* representation discovers. Section 10.43.9 runs exactly that check, and the answer is three-way: an unbounded numerically-optimised constant range (PySR) misses the bound at the published 40-iteration budget but *discovers* it at 120, recovers the gravity gate that gplearn's drawn constants miss, and admitting the clamp as a candidate structure recovers the bound at any budget. gplearn itself was then swept over a 48x range of evaluations and also discovers the bound at its top - while its held-out accuracy falls at exactly that point - so neither engine is blocked, and what the comparison measures is the price at which each buys structure. (ii) Bagging over 3 seeds is variance reduction, not a posterior; no uncertainty statement about a discovered expression is claimed. (iii) The synthetic hidden world is the repository's own analytic generator, so its laws are piecewise-affine by construction - favourable to this search, unlike a table-driven console acceleration curve. (iv) GP seeds are fixed, but the search is not order-invariant: 3 seeds per law is the minimum honest sample, not a converged distribution, which is why every structural claim in this section is reported as a *rate*. (v) The S1/S2 comparison refits the parametric estimator at 900 steps against 10.40-E1's published 2000, so the "Parametric ID" column here is a slightly weaker version of the published one (S4: 0.32 vs 0.28 px); the ordering is unaffected. (vi) The real-data rows give both inverse models the terrain-contact byte as input, as 10.40-E2 does; the learned baselines do not get that favour and are therefore not comparable on $x$ or $y$.

![Symbolic regression for the inverse problem](../../results/figures/symbolic_inverse_recovery.png)

*Left:* per-constant recovery error (symlog) for the prior, the symbolic GP + probe estimator and the 10.40 parametric identification - the $\max_{v_x}$ bar is absent for GP because no fixed point was discovered. Centre: held-out R² of the two velocity laws against the GP budget, the control that separates representation limits from search effort. Right: per-channel test MSE on genuine WRAM telemetry.

Regenerate: `python -m src.evaluation.symbolic_inverse_benchmark` (emulator-free, ~20 min on CPU; `--replicates`, `--gp-seeds`, `--population-size`, `--generations`, `--max-train`, `--weight-strengths` and `--no-budget-sweep` control the search budget; `make symbolic-inverse` / `smw-pinn symbolic-inverse`, smoke config `configs/smoke_symbolic_inverse.yaml`).

---

#### 10.43.8 The Counterfactual: Hand the Same Search the Terrain It Was Missing

Section 10.43.5 ended on an attribution: the identified analytic model's horizontal-velocity residual has negative *train* R² even with all eleven observable channels, so it "is tile geometry and slope acceleration that never enter the observation". That is an inference from an absence of evidence - the state vector does not carry terrain, and the residual does not yield. The repository, however, already holds the counterfactual recording: `smw_tilemap_dataset.npz` (Section 10.20) records the $7 \times 7$ local WRAM block buffer around Mario, tile by tile, for every transition of the same stage. So the claim is testable, and testing it is the difference between a diagnosis and a guess.

**Design.** The analytic map is re-identified on the tilemap recording's own training split (3,489 fit transitions at a stride of 2, 2,004 test transitions; its constants land 0.0-74.3% from the reverse-engineered values, with $\sigma$ at 39.4%), its per-channel residual becomes the regression target, and genetic programming is asked to explain that residual under five conditioning conditions that differ *only* in what the search may see:

| Condition | Features | What is added |
| :--- | :---: | :--- |
| `state` | 11 | the 10.43.5 replication on this recording |
| `state+geom` | 18 | seven occupancy descriptors of the patch (below, ahead-right, ahead-left, ceiling, center, slope fraction, fill) |
| `state+geom+inter` | 22 | plus four terrain × input products (`dir` facing-side, `vx` into the faced wall, `vy` into the floor) |
| `state+full` | 60 | all 49 raw patch cells, row 0 above Mario and column 0 to his left |
| `state+geom-shuffled` | 22 | the descriptors **permuted across frames** - a placebo that must not help |

Every GP row is reported as the median over 3 seeds (with best and worst), and beside it an ordinary least-squares fit on the identical design matrix, so a GP failure can be told apart from "there is nothing linear here". Support is granted only when the geometry condition predicts out of sample ($R^2 \gt 0.05$), beats the state-only fit by more than 0.05, *and* beats the placebo by more than 0.02 - "less catastrophic" is not "explained".

**Result: the terrain explains the vertical residual and not the horizontal one.**

| Residual | `state` median | best geometry condition | median there | gain | placebo gain | max $\lvert\text{corr with terrain}\rvert$ | terrain explains it |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $x$ | +0.667 | `state+geom+inter` | +0.677 | +0.010 | +0.010 | 0.138 | **no** |
| $y$ | +0.657 | `state+geom+inter` | +0.735 | +0.077 | +0.092 | 0.181 | **no** (placebo wins) |
| $v_x$ | -0.011 | `state+geom` | -0.011 | +0.000 | +0.000 | 0.068 | **no** |
| $v_y$ | +0.311 | `state+geom+inter` | +0.365 | +0.054 | -0.007 | 0.243 | **yes** |

Three findings, one of them a correction of this section's own predecessor:

1. **The $v_y$ residual really is terrain.** Handing the search occupancy raises held-out R² from +0.311 to +0.365 while the shuffled placebo *lowers* it (-0.007), and the strongest single terrain correlation in the whole table is ceiling-above versus the vertical-velocity residual (0.243). The best discovered expression is itself a physical statement - `mul(max(mul(patch_fill, vy_x_below), vy), ground)` - a ground-gated reset of downward velocity built from the floor occupancy rather than from the contact byte. The analytic model's vertical error is therefore not merely "crudely approximated collision response" (10.40-E2's wording): it is a function of terrain the state vector omits but the block buffer carries.
2. **The $v_x$ attribution in 10.43.5 was unsupported, and is retracted.** The horizontal residual does not move at all: gain exactly +0.000 in every condition, placebo identical, and no terrain descriptor correlating above 0.068 with it. Local block occupancy is *not* the missing variable for horizontal velocity. What the test can say positively is narrower and less comfortable: within this recording, the horizontal residual is explained by neither the state nor the terrain around Mario. One candidate survives only because the recording cannot express it - the slope class (tile code 3) never occurs in this stage's patches, so slope acceleration is untestable here by construction - and the others (sub-tile collision resolution, sprite interactions, telemetry quantisation) are outside both observation sets. Section 10.43.5's sentence has been corrected accordingly, and this is recorded as a self-correction in Section 12.
3. **The placebo earned its keep.** The $y$ residual shows a +0.077 gain that would have been reported as terrain-explained by any design without the shuffled condition - and the placebo scores higher still (+0.092), which identifies it as overfitting to the 22-feature design rather than as physics. Two of the four apparent "explanations" in this table exist only because the control was run.

A secondary observation, unfavourable to the method and worth stating: on the $v_x$ residual, plain least squares on the eleven state features reaches +0.081 out of sample while genetic programming sits at -0.011 (and one of its three seeds collapses to -8.07 by extrapolating an unbounded expression). Symbolic search is not uniformly better than a linear fit on noisy telemetry; it buys interpretability and structure, and on this channel it pays for them with accuracy.

![Tilemap-conditioned residual discovery](../../results/figures/symbolic_tilemap_residual_gain.png)

*Held-out R² on the identified analytic model's residuals, per channel, for each conditioning condition: genetic programming (median over seeds, with best/worst whiskers) against least squares on the same design. The rightmost group in each panel is the shuffled-geometry placebo.*

Regenerate: `python -m src.evaluation.symbolic_tilemap_residual_benchmark` (emulator-free, ~6 min CPU; `make symbolic-tilemap`, smoke config `configs/smoke_symbolic_tilemap.yaml`; `--stride`, `--gp-seeds`, `--population-size`, `--generations` and `--id-steps` control the budget).

#### 10.43.9 Is It the Representation or the Criterion? Three Engines on the Same Law

Section 10.43 reported that genetic programming failed to discover the rigid velocity ceiling in 27 draws and attributed it to the estimator's terminals - "gplearn draws its constants uniformly from $[-4,4]$, so a bound at 48 is not expressible" - with the caveat that this was a statement about one representation. The caveat is testable. `src/evaluation/symbolic_engine_ablation_benchmark.py` runs the *same* inverse problem through three engines of increasing structural generosity and asks each the same structural questions:

* **gplearn tree GP** - the published 10.43 estimator: free-form trees, terminal constants drawn from $[-4,4]$, mean-error fitness.
* **PySR** - the same kind of tree search, but with constants refined by numerical optimisation (unbounded, so 48 is representable) and `min`/`max` as first-class operators. Optional dependency (`pip install -e ".[symbolic]"`, it drags in a Julia runtime); when absent the row is recorded as unavailable, never dropped. Run `parallelism="serial", deterministic=True` so the published artifact is reproducible.
* **Nested templates + model selection** (`src/inverse/structure_selection.py`) - not a search: six horizontal and six vertical structures, from `H1_constant` up through the traction tiers, the rigid clamp, the Coulomb coast deadband and the contact stop, each an explicit candidate, scored by BIC on the fit rows, by held-out RMSE, and by held-out RMSE restricted to the near-bound frames.

What is held fixed is the point of the exercise: identical design matrices and feature presets, identical row budget ($\leq 4000$ fit transitions), identical held-out bank, identical probes (the bound is a fixed point of the driven map, rejected when the map never accelerates; the gravity gate is the median tier separation), and accuracy scored on the shared bank for all three. 3 replicates × 3 GP seeds on the 10.40-E1 hidden world, then the same three engines on genuine WRAM telemetry.

| Engine (hidden world, 3 replicates) | $\Delta v_x$ held-out R² | bound discovered | bound picked by BIC | recovered bound | gate discovered | measured tier separation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| gplearn tree GP (as published) | 0.805 | **0.00** | 0.00 | - | 0.33 | -1.35 |
| PySR (optimised unbounded constants) | 0.988 | **0.00** | 0.00 | - | 1.00 | -2.80 |
| Nested templates + selection | 1.000 | **1.00** | 1.00 | 48.000 | 1.00 | -2.80 |
| *Exact engine map (reference)* | 1.000 | - | - | 48.000 | - | -2.80 |

1. **The ceiling is in the data. It was only ever missing from the search.** Offered as a candidate structure, the clamp is recovered at 48.000 against a true 48.0 and selected by BIC *and* both held-out criteria in 3 of 3 replicates (the horizontal winner is `H5_coast`, bound plus Coulomb deadband; the vertical winner `V6_ground_reset`, gate plus clamp plus landing reset; `criteria_agree` is true on all six runs), and the recovered constants sit between $4\times10^{-6}\%$ and $1.4\times10^{-4}\%$ from truth ($v_{\max}$ 48.0, $\tau_{\text{walk}}$ 1.0, $\tau_{\text{run}}$ 1.8, deceleration 0.6, $g_{\text{hold}}$ 2.4, $g_{\text{fall}}$ 5.2). That closes the alternative reading of 10.43 - "these windows do not excite the constraint enough to identify it". They do.
2. **At the published budget, the bound is not a representation problem and not an accuracy problem either.** Neither free tree search finds it - 0 of 3 bagged replicates for gplearn, 0 of 3 fits for PySR - and the engine that misses it is not the worse-fitting one: PySR's held-out R² on the shared bank is 0.988 against gplearn's 0.805, its constants are unbounded, and it still returns no fixed point. What its law does instead is instructive - it is a `min`/`max`-clamped expression whose driven map stays within 0.213 px/frame of the observed velocity support *without ever turning over inside it*. An aggregate error metric prices that near-miss at essentially nothing - a law with R² 0.988 is not being penalised for the constraint - and the artifact records where a clamp could be seen at all: it binds on 27.1% of held-out synthetic frames and on 0.0% of the telemetry test split, because the recorded play never revisits 90% of the training speed support. The template engine is the only one of the three that scores any structure on the binding frames separately, and it is the only one that finds the bound. **This finding does not survive a larger budget** (finding 6), and Section 10.45 shows it does not survive better-excited data either.
3. **The gravity gate, by contrast, really was a representation limit.** The held-jump tier step is -2.80 px/frame: gplearn approximates it to -1.35 and recovers it as a gate in 1 of 3 replicates, while PySR - same operators in kind, but constants fitted numerically rather than drawn - reproduces -2.80 and recovers the gate in 3 of 3. Two structures that both defeated 10.43 therefore fail for two different reasons, and only one of them is about the terminal range.
4. **On genuine telemetry the ordering survives, and a new trap appears.** The template engine fits the horizontal law at held-out R² 0.989 and reports a clamp at 47.775 - 33.6% from the WRAM reference 72.0, and just under the 49.0 this recording actually reaches: with no frames beyond the support to constrain it, the fit parks the ceiling at the edge of the data rather than at the engine's limit. PySR reaches 0.430 and reports a bound at 34.193, a ceiling the *data itself violates* (frames at 49.0 exist), yet the fixed-point probe accepts it: a probe that only looks for a turnover below the explored range cannot tell a constraint from a mis-fit. gplearn reproduces 10.43.5's $-1.7\times10^{-4}$ to the digit - no better than the mean predictor - and reports no bound at all. None of the three recovers the gravity gate from telemetry, and inside the template engine BIC and the tail criterion disagree ($H6_{\text{contact}}$ vs $H5_{\text{coast}}$), so on real data the selection criterion still decides which law is reported even when every candidate carries the constraint. "A probe found a fixed point" is therefore not the same statement as "the engine has this ceiling", and the numbers here are reported with that distinction open.

5. **The selector can name a mechanism it was not authored from.** The obvious objection to a dictionary written after reading 10.37 is that it will confirm what it was given, so the claim was tested rather than defended: `fit_ceiling_families` fits four *mutually exclusive* explanations of "the speed stops growing" - no ceiling at all, a rigid clamp, a quadratic drag whose asymptote is the level, and an exponential relaxation toward it - to the same driven rows, on data generated by each of the three plateau mechanisms in turn. The selector names the true mechanism in 3 of 3 controls with BIC, held-out RMSE and the tail criterion agreeing every time, and it separates drag from clamp on the signature that distinguishes them (drag has two asymptotes, one per traction tier; a clamp has one). The dictionary is therefore a discriminator rather than a rubber stamp, which is what licenses finding 1 as a statement about the data and not about its author.
6. **Triple the search budget and the free search finds the whole law - which is the strongest result in this section and a correction to its own conclusion.** The same replicate, the same data and the same seed, re-run at 120 PySR iterations instead of 40, returns a fixed point: held-out R² goes $0.9983 \to 1.0000$ and the driven map's overshoot of the velocity support goes $0.0014 \to 0.0$ px/frame (the two fits cost 633 s and 1030 s of serial search on this host, and that ratio is machine-dependent while the iteration count is not). The stored expression is not an approximation of a ceiling, it *is* the ceiling - evaluated on the driven branch it is exactly $\hat v = \min(v + 1.8,\;48.0)$, whose fixed point is 48.0 against a true 48.0, with the walk tier at 1.0, the run tier at 1.8 and the Coulomb deadband at 0.5999992 against a true 0.6. So the structural discovery that 10.43 reported as impossible, and that finding 2 of this section attributed to the fitness, is achieved by a tree search given three times the iterations: the honest statement is that discovery here is **budget-limited**, not that it is blocked.

7. **gplearn flips too - at sixteen times the budget, and only by *sacrificing* accuracy.** The published limitations named this measurement as the obvious missing one, so it was run: population × generations over $500\times25 \to 2000\times300$, a 48x increase in evaluations, deliberately starting where 10.43's S1b left off and reaching far past it, with the row budget pinned at 4000 transitions so only effort varies. The fixed point is absent at $500\times25$, $500\times75$ and $1000\times150$ and appears at $2000\times300$. Two things are worth separating here. The first is the asymmetry, which is the quantitative content of the section: PySR needed 3× its iterations to discover the bound, gplearn needed 48× its evaluations - the two tree engines differ not in whether effort buys structure but in how much effort costs it. The second is the sharper one:

   | gplearn budget | evaluations | held-out R² | fixed point | mean nodes |
   | :--- | :---: | :---: | :---: | :---: |
   | $500\times25$ (published) | 12,500 | 0.788 | no | 5.8 |
   | $500\times75$ | 37,500 | 0.882 | no | 5.5 |
   | $1000\times150$ | 150,000 | **0.926** | no | 10.8 |
   | $2000\times300$ | 600,000 | 0.907 | **yes** | 10.8 |

   Held-out accuracy *falls* from 0.926 to 0.907 at the moment the constraint appears. Under a mean-error fitness, discovering the bound costs accuracy - which is exactly the mechanism finding 2 hypothesised, now measured instead of argued: the search is not failing to see the constraint, it is being paid not to buy it. (The gravity gate reads yes at every budget in this sweep because it reuses replicate 0, whose vertical law separated the tiers; the published matrix gate rate of $1/3$ averages all three replicates.)

![Three-engine discovery ablation](../../results/figures/symbolic_engine_ablation.png)

*Rate at which each engine discovers the rigid velocity bound (left) and the held-jump gravity gate (right) on the hidden world.*

**Limitations, the largest first.** The template repertoire was authored by someone who had already read Section 10.37's reverse-engineered engine rules, so this is a comparison between *searching* a physics-authored dictionary and *not* searching one; it says nothing about whether an unaided search would have written that dictionary, and it inherits the hand-designed-model caveat of 10.37. Finding 5 tests that objection directly and the selector passes it, but a pass on three synthetic plateau mechanisms is still not a claim that the dictionary would have been written without the engine rules. PySR ran serially for 40 iterations per law in the published rows to keep the artifact reproducible - and finding 6 shows that this choice was load-bearing rather than cosmetic, since the structural answer flips at 120; finding 7 sweeps gplearn over the same kind of range and it flips as well, at 48× evaluations and at a measurable cost in held-out accuracy. Both tree engines are therefore budget-limited here, and what the section can still claim is the *rate* at which each buys structure, not a qualitative block. Three replicates is enough to separate $0/3$ from $3/3$ and not enough to resolve anything between. The probes are the same for every engine, which makes them a shared assumption rather than a shared ground truth: a structure that binds outside the probed range, or only in combination with an unmodelled channel, is invisible to all three.

Regenerate: `python -m src.evaluation.symbolic_engine_ablation_benchmark` (emulator-free, ~2 h CPU with the PySR legs and both budget sweeps - the gplearn grid alone is ~25 min of it, most of that in the $2000\times300$ fit; `make symbolic-engines`, smoke config `configs/smoke_symbolic_engines.yaml`, which pins both sweeps to token budgets; the PySR legs are skipped-and-recorded unless the `symbolic` extra is installed; `--replicates`, `--gp-seeds`, `--pysr-iterations`, `--pysr-budgets`, `--gplearn-effort-grid`, `--specificity-rows`, `--specificity-noise` and `--max-train` control the budget).
