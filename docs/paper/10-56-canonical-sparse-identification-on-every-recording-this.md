[← 10.55](10-55-which-numerical-method-does-the-engine-use.md) · [Contents](../../README.md#contents) · [10.57 →](10-57-what-section-4-costs-the-corrected-physics.md)

### 10.56 Canonical Sparse Identification on Every Recording This Repository Has

Everything this repository has claimed about *recovering* the laws came from instruments built for other questions: §10.43's evolutionary program search (gplearn, with the PySR leg recorded as skipped because the Julia runtime does not start on this machine) searches expression trees over a function set, and §10.37's gradient identification fits the constants of a law whose structure a human wrote down first. Neither is the canonical estimator of the SciML literature. `src/inverse/sindy_identification.py` is: polynomial dictionary, ridge regression with the constant column exempt, sequential thresholding iterated to a sparse support (SINDy / STLSPL, Brunton, Proctor and Kutz 2016), run here as the *forward* problem - identify the dynamics from telemetry with no physics in the loop, no architecture and no emulator, in the twenty seconds a dictionary fit costs rather than the fourteen minutes 10.53's grid does.

Protocol, written out because two of the findings below are about it: five recordings (the published gameplay set, §10.31 multi-entity, §10.41 tilemap, §10.45 sprint-targeted, §10.52 jump-targeted); degree-2 dictionary over the twelve non-position channels (velocity, contacts, actions) - positions are deliberately excluded, so the identification cannot see level geometry; 81 columns before pruning; the derivative estimated as the forward difference over one frame, which makes a $\Delta x$ coefficient pixels per frame and a $\Delta v$ coefficient sub-pixels per frame per frame; thresholding at $\alpha = 0.05$ with a sweep of (0.5, 0.2, 0.1, 0.05, 0.02, 0.01) reported alongside; and two losses, plain least squares and a Huber IRLS variant. Columns are standardised for conditioning *only* and rescaled before thresholding, so $\alpha$ is in physical units - a claim that needs its own test, and has one.

| recording | loss | terms $\Delta x$ / $\Delta v_x$ / $\Delta v_y$ | scale on $v_x$ ($\Delta x$) | constant in $\Delta v_y$ | jump in $\Delta v_y$ | rel. err. $\Delta x$ | R² $\Delta v_x$ | R² $\Delta v_y$ |
|:--|:--|:--|--:|--:|:--:|--:|--:|--:|
| published gameplay | `least_squares` | 12 / 21 / 21 | 18.00 | 1.134 | no | 0.2595 | 0.084 | 0.005 |
| published gameplay | `robust` | 5 / 23 / 6 | 16.00 | 3.000 | no | 0.2872 | 0.124 | 0.006 |
| tilemap (10.41) | `least_squares` | 16 / 24 / 34 | 19.01 | 1.769 | **yes** | 0.1981 | -0.041 | 0.087 |
| tilemap (10.41) | `robust` | 6 / 26 / 13 | 16.00 | 5.995 | **yes** | 0.1921 | -0.076 | -0.021 |
| multi-entity (10.31) | `least_squares` | 11 / 23 / 32 | 16.17 | 3.072 | **yes** | 0.2105 | 0.181 | 0.279 |
| multi-entity (10.31) | `robust` | 6 / 25 / 15 | 16.00 | 5.989 | **yes** | 0.2123 | 0.167 | 0.104 |
| sprint-targeted (10.45) | `least_squares` | 8 / 19 / 24 | 18.73 | 3.744 | **yes** | 0.2753 | 0.054 | 0.171 |
| sprint-targeted (10.45) | `robust` | 3 / 19 / 16 | 16.00 | 3.740 | **yes** | 0.2881 | 0.049 | 0.112 |
| jump-targeted (10.52) | `least_squares` | 8 / 16 / 21 | 18.27 | 2.297 | **yes** | 0.1024 | 0.156 | 0.482 |
| jump-targeted (10.52) | `robust` | 3 / 17 / 19 | 16.00 | 3.764 | **yes** | 0.0066 | 0.144 | 0.455 |

The same constants read three ways - this study on the published recording, §10.37's gradient fit of the hand-written structure, and what §4 documents:

| quantity | SINDy, least squares | SINDy, Huber | §10.37 gradient | §4 documented |
|:--|--:|--:|--:|--:|
| sub-pixels per pixel ($\Delta x$ on $v_x$) | 18.00 | 16.00 | 21.41 | 16 |
| ascent gravity (constant in $\Delta v_y$) | 1.134 | 3.000 | 1.325 | 3 |
| ground coefficient in $\Delta v_y$ | -0.604 | -3.000 | - | not documented |
| run-button coefficient in $\Delta v_x$ | - | +0.294 | 0.515 | 1.5 |
| coast coefficient on $v_x$ in $\Delta v_x$ | - | term absent | 0.445 | 0.5 |

The supports themselves, for the two recordings whose laws finding 3 and finding 5 read - the published set and the one collected to contain the gate (the eighth-largest term is shown, and the row says how many more there are):

| recording | target | the discovered support (Huber, $\alpha = 0.05$) | terms | rel. err. | R² |
|:--|:--|:--|--:|--:|--:|
| published gameplay | $dx$ | 0.4509·c_right a_right + -0.2255·c_right + -0.2254·c_ground c_right + -0.2254·c_right a_jump + 0.0625·vx | 5 | 0.2872 | 0.907 |
| published gameplay | $dvx$ | 3.2050·c_left a_right + 2.7900·c_right a_right + -1.9644·a_run a_left + -1.0000·vx c_right + -0.9479·vx c_left + -0.9473·c_right a_run + 0.6538·a_run a_right + -0.6358·a_left + … (15 more terms) | 23 | 0.9357 | 0.124 |
| published gameplay | $dvy$ | 2.9999 + -2.9999·c_ground + -2.9211·a_left + 2.9211·c_ground a_left + -0.9454·a_down + 0.9454·c_ground a_down | 6 | 0.9970 | 0.006 |
| jump-targeted (10.52) | $dx$ | 0.0708·c_ground c_right + -0.0708·c_ground a_right + 0.0625·vx | 3 | 0.0066 | 0.999 |
| jump-targeted (10.52) | $dvx$ | -1.7578·a_left + 1.7578·a_right + 1.7525 + -1.4174·c_right a_run + -0.7574·c_right + -0.5865·vx c_right + 0.5719·a_run a_right + 0.5337·a_run + … (9 more terms) | 17 | 0.9254 | 0.144 |
| jump-targeted (10.52) | $dvy$ | 39.0074·c_ground a_left + -38.6002·c_ground a_right + -21.4052·c_ground c_right + -10.8739·c_ground + -7.9559·c_right + 5.4974·c_right a_run + -4.3902·a_jump a_right + 3.7637 + … (11 more terms) | 19 | 0.7385 | 0.455 |

1. **The integration constant is exactly recoverable from data alone, but only by the robust loss.** The Huber fit puts the $\Delta x$ coefficient on $v_x$ at 16.000009, 16.000094, 16.000001, 16.000027 and 16.000020 on the five recordings - i.e. $1/16$ px per sub-pixel, the engine's fixed-point scale, to within $9.4 \times 10^{-5}$ of it. Plain least squares on the same recordings and the same dictionary gives 16.17, 18.00, 18.27, 18.73 and 19.01, and §10.37's gradient identification of the same published recording gave 21.41. That closes 10.37's number: the 21.41 was not a second scale constant in the engine, it was an estimator letting a handful of collision and wrap frames set the slope. Three instruments on one recording, 16.0 / 18.0 / 21.4, is a statement about loss functions, not about Mario.
2. **The gravity gate survives thresholding at 79-98% of its measured size under the robust loss, and its sign is wrong under least squares.** Four of the five recordings contain released-ascent frames; §10.52 measured their tier separation as exactly +3.0 sub-pixels/frame (3.0 held, 6.0 released), and the identified laws reproduce +2.935 (tilemap), +2.891 (multi-entity), +2.543 (sprint) and +2.378 (jump). The least-squares fits of the same four recordings give -0.746, -1.411, -0.368 and -1.163: they put the *held* tier at the stronger gravity, which is the documented rule backwards. The fifth recording is the control in the other direction - it contains no released ascent (measured separation +0.0), and the robust fit invents none (+0.000), so the spurious-gate list in the artifact is empty. This is 10.53's lesson in a different instrument: a robust loss is not a refinement here, it is the difference between recovering a law and recovering its negation.

| recording | loss | held ascent: frames, observed median $\Delta v_y$, law's mean | released: same | measured separation | identified separation |
|:--|:--|:--|:--|--:|--:|
| published gameplay | `least_squares` | 82, 3.0, law 1.95 | 72, 3.0, law 1.68 | +0.0 | -0.268 |
| published gameplay | `robust` | 82, 3.0, law 3.00 | 72, 3.0, law 3.00 | +0.0 | +0.000 |
| tilemap (10.41) | `least_squares` | 208, 3.0, law 3.97 | 247, 6.0, law 3.23 | +3.0 | -0.746 |
| tilemap (10.41) | `robust` | 208, 3.0, law 3.00 | 247, 6.0, law 5.93 | +3.0 | +2.935 |
| multi-entity (10.31) | `least_squares` | 760, 3.0, law 4.34 | 514, 6.0, law 2.93 | +3.0 | -1.411 |
| multi-entity (10.31) | `robust` | 760, 3.0, law 3.00 | 514, 6.0, law 5.89 | +3.0 | +2.891 |
| sprint-targeted (10.45) | `least_squares` | 374, 3.0, law 3.31 | 440, 6.0, law 2.94 | +3.0 | -0.368 |
| sprint-targeted (10.45) | `robust` | 374, 3.0, law 2.98 | 440, 6.0, law 5.52 | +3.0 | +2.543 |
| jump-targeted (10.52) | `least_squares` | 325, 3.0, law 2.78 | 338, 6.0, law 1.61 | +3.0 | -1.163 |
| jump-targeted (10.52) | `robust` | 325, 3.0, law 2.93 | 338, 6.0, law 5.31 | +3.0 | +2.378 |

3. **The vertical law identifies the ground, not just the button.** On the published recording the robust fit is $\Delta v_y = 2.9999 - 2.9999\,c_{\text{ground}} - 2.9211\,a_{\text{left}} + 2.9211\,c_{\text{ground}}a_{\text{left}} - 0.9454\,a_{\text{down}} + 0.9454\,c_{\text{ground}}a_{\text{down}}$ - six terms, and the first two are the +3.0 of §4.2.3 and an exact -3.0 cancellation when the ground flag is set. That second number is not in §4 and not in any shell in this repository: §10.54 measured that the recorded $v_y$ is *never* zero on grounded frames (median $|v_y| = 6.0$ over 2,943 of them), and here is the same fact as an identified coefficient - the ground flag in `$7E:0077` gates the gravity *application*, and the constant $g$ the models are penalised against is what is left when it does not. The remaining four terms are a single airborne projection ($-2.9211\,a_{\text{left}}(1-c_{\text{ground}})$ plus the same in $a_{\text{down}}$): correlated-button structure in one recording, not a law, and it is reported rather than smoothed over.
4. **The threshold is a units question, and the pooled score does not flag getting it wrong.** $\alpha$ is in physical units, and $1/16 = 0.0625$ px/frame is the coefficient the engine's integration rule lives at, so any threshold above it deletes the law. At $\alpha = 0.02$ the jump-targeted $\Delta x$ support is exactly one term, $0.0625\,v_x$, with held-out relative error $1.8 \times 10^{-6}$ and $R^2 = 0.9999999999$ - the integration rule recovered with nothing but data; at 0.05 the same term is there with two contact partners (relative error 0.0066). At 0.1 and coarser it is gone: on the published recording the whole $\Delta x$ law at $\alpha = 0.1$ is $1.6191 - 1.3854\,c_{\text{right}}a_{\text{jump}} - 0.7693\,c_{\text{ground}} - 0.7059\,c_{\text{right}} - 0.6438\,c_{\text{ground}}c_{\text{right}} + 0.5972\,c_{\text{right}}a_{\text{right}}$ (no velocity term, relative error 0.8612), and at $\alpha = 0.5$ it is a single wall term with relative error 1.0000 and R² -0.1307. What makes this a protocol finding rather than a detail is that the *pooled* held-out R² improves across exactly that range - 0.0040, 0.0090, 0.0169 at $\alpha$ = 0.5, 0.2, 0.1 - because the velocity channels dominate it, while the $\Delta x$ column is at chance. A sweep scored on the pooled loss stops early and calls the deletion an improvement; the per-target curve is the only one that sees the law it lost. Below 0.05 the published $\Delta x$ terms are unchanged and its relative error is flat at 0.2872, so the selected threshold is where the law appears, not where the score peaks.

| recording | $\alpha$ | terms in $\Delta x$ | the $\Delta x$ support (Huber) | rel. err. $\Delta x$ | R² $\Delta x$ | pooled R² |
|:--|--:|--:|:--|--:|--:|--:|
| published gameplay | 0.5 | 1 | 0.1000·c_right a_right | 1.0000 | -0.1307 | 0.0040 |
| published gameplay | 0.2 | 4 | 0.5972·c_right a_right + -0.2511·c_ground c_right + -0.2489·c_right + -0.2233·c_right a_jump | 1.0001 | -0.1310 | 0.0090 |
| published gameplay | 0.1 | 6 | 1.6191 + -1.3854·c_right a_jump + -0.7693·c_ground + -0.7059·c_right + -0.6438·c_ground c_right + 0.5972·c_right a_right | 0.8612 | 0.1615 | 0.0169 |
| published gameplay | 0.05 | 5 | 0.4509·c_right a_right + -0.2255·c_right + -0.2254·c_ground c_right + -0.2254·c_right a_jump + 0.0625·vx | 0.2872 | 0.9068 | 0.0366 |
| published gameplay | 0.02 | 5 | 0.4509·c_right a_right + -0.2255·c_right + -0.2254·c_ground c_right + -0.2254·c_right a_jump + 0.0625·vx | 0.2872 | 0.9068 | 0.0603 |
| published gameplay | 0.01 | 5 | 0.4509·c_right a_right + -0.2255·c_right + -0.2254·c_ground c_right + -0.2254·c_right a_jump + 0.0625·vx | 0.2872 | 0.9068 | 0.0598 |
| jump-targeted (10.52) | 0.5 | 1 | 2.0014·c_ground a_right | 0.9797 | -17.2608 | 0.4369 |
| jump-targeted (10.52) | 0.2 | 1 | 2.2198·a_right | 0.1062 | 0.7853 | 0.4498 |
| jump-targeted (10.52) | 0.1 | 4 | 1.1107 + -1.1107·a_left + 1.1107·a_right + -0.2233·c_ground a_right | 0.1069 | 0.7825 | 0.4511 |
| jump-targeted (10.52) | 0.05 | 3 | 0.0708·c_ground c_right + -0.0708·c_ground a_right + 0.0625·vx | 0.0066 | 0.9992 | 0.4514 |
| jump-targeted (10.52) | 0.02 | 1 | 0.0625·vx | 0.0000 | 1.0000 | 0.4688 |
| jump-targeted (10.52) | 0.01 | 1 | 0.0625·vx | 0.0000 | 1.0000 | 0.5095 |

5. **What SINDy does not recover here is the horizontal velocity law, and the reason is structural.** The published recording's $\Delta v_x$ support is 23 terms with relative error 0.9357 and R² 0.124; the jump recording's is 17 terms at 0.9254. §4.3.4's "friction" has no single coefficient in this data: the run button enters as +0.294 on its own but also as $+3.205\,c_{\text{left}}a_{\text{right}}$, $+2.790\,c_{\text{right}}a_{\text{right}}$, $-0.947\,c_{\text{right}}a_{\text{run}}$ and $+0.654\,a_{\text{run}}a_{\text{right}}$, while the bare coast term on $v_x$ does not survive thresholding at all. Deceleration in this engine is surface-conditioned - which wall you are touching, which direction you are pressing against it - and a degree-2 polynomial dictionary on the instantaneous state can only represent that as an interaction pile. The same reading explains the vertical channel's held-out R² of 0.006 on the published recording: the constants are right and the frame-by-frame residual is not, because the exceptions the constants do not describe are what the variance is made of. That is precisely the gap 10.53's `offset` head was built to buy, and here is an instrument with no head to buy it with.
6. **The dictionary is rank-limited by the recording, and the pruning step measures by how much.** Of the 81 degree-2 columns, 36 of 81 are constant or exact duplicates on the published recording and 45 survive; the other four recordings keep 46 (tilemap), 48 (multi-entity), 32 (sprint) and 31 (jump). The dropped list is a finding rather than plumbing: `c_ceiling` and `a_up` are constant in the published recording (so every product containing them is constant), and the exact duplicates include `c_ground c_left ≡ c_left`, `a_down a_right ≡ a_down`, and on the sprint and jump recordings `vx a_right ≡ vx` - the policy held right so long that velocity and "holding right" are the same column. No identification on these recordings, symbolic or otherwise, can separate those structures, which retroactively bounds what §10.43's search could have found.

**Limitations.** These are increments of a discrete map, not derivatives of a continuous one: the forward difference over one frame is what the engine's own 60 Hz step is, so the exact $0.0625\,v_x$ recovered above is the fixed-point convention rather than a physical law, and reading this section as "SINDy recovered Newtonian mechanics" would be wrong - it recovered the arithmetic. The ridge and threshold are single-shot on one split (seed 42), so unlike 10.53 there is no five-seed spread behind these numbers; the $\alpha$ sweep is the sensitivity analysis offered instead. Degree 2 is the whole dictionary, and the inputs exclude positions, the P-meter and the debounced button latch, so anything depending on those is invisible by construction. The five recordings share the canonical episodic split, so "5 recordings" is 5 policies and 5 stage sets, not 5 independent engines. Finally, an identified law that predicts well is not a usable world model: the best pooled held-out R² in this study is 0.45 on the jump recording and 0.04 on the published one, while 10.53's `carried` arms reach 0.1056 px of position error and fly the console - identification and prediction are different jobs, and this leg is reported as evidence about the constants and the estimators, not as a competitor to the grid.

![Sparse identification of the telemetry](../../results/figures/sindy_identification.png)

*Left:* held-out R² per target, per recording, solid for the Huber fit and dashed for least squares - the $\Delta x$ column is the only one that is consistently good. Right: the threshold sweep, surviving terms against pooled held-out R², one curve per recording; the curves rise monotonically toward density and never turn down where the $\Delta x$ law loses its $v_x$ term, which is finding 4.

Regenerate: `python -m src.evaluation.sindy_identification_benchmark --alpha 0.05` (emulator-free, ~20 s for the five recordings, writes `results/sindy_identification_metrics.json`), or `make sindy`.

#### 10.56.1 Conformal Intervals on the Identified Laws

Section 10.43 publishes the seed-to-seed spread of a discovered law as the honest substitute for the Bayesian uncertainty it cannot compute - a regression tree has no Hessian to invert, so the Laplace machinery of 10.40 has no symbolic analogue. Split conformal prediction (Vovk, Gammerman and Shafer 2005; the split variant of Papadopoulos et al. 2002) needs neither a functional form nor a gradient: score the fitted law on a calibration split the fit never saw, read the half-width off the ceil((n+1)(1-alpha)) order statistic, and the marginal coverage of at least 1 - alpha is a finite-sample statement that assumes nothing about the residual distribution. `src/inverse/conformal_uncertainty.py` implements the three lines of arithmetic; `src/evaluation/conformal_uncertainty_benchmark.py` applies them to the 10.56 laws exactly as that section fits them (Huber loss, threshold 0.05, degree-2 dictionary, duplicates pruned), on all five recordings, with the canonical validation episodes as calibration and the canonical test episodes as evaluation - the same splits every model in 10.47/10.53/10.55 was trained and scored on.

Three arms per cell. The **law arm** centres the interval on the identified law's own prediction. The **Gaussian arm** reads z times the calibration sigma instead - the parametric statement 10.58 already measured under-predicting this telemetry's tails. The **fit-mean null** runs the same conformal machinery around the calibration mean, the control every per-law table in this repository carries: a law whose interval is no tighter than the mean's has earned nothing. Coverage at the 90% level, one row per recording and target:

| recording | target | conformal coverage | conformal half-width | Gaussian coverage | null coverage |
| :-- | :--: | :--: | :--: | :--: | :--: |
| published gameplay | dx | 1.000 | 2.1875 | 0.970 | 0.774 |
| published gameplay | dy | 1.000 | 6.9553 | 0.941 | 0.923 |
| published gameplay | dvx | 0.915 | 1.6902 | 0.986 | 0.957 |
| published gameplay | dvy | 0.979 | 6.0000 | 0.988 | 0.988 |
| tilemap (10.41) | dx | 0.939 | 0.0000 | 0.940 | 0.904 |
| tilemap (10.41) | dy | 0.948 | 0.9910 | 0.958 | 0.936 |
| tilemap (10.41) | dvx | 0.901 | 1.8324 | 0.972 | 0.914 |
| tilemap (10.41) | dvy | 0.922 | 5.9946 | 0.977 | 0.977 |
| multi-entity (10.31) | dx | 0.901 | 0.0000 | 0.960 | 0.937 |
| multi-entity (10.31) | dy | 0.876 | 0.9731 | 0.853 | 0.929 |
| multi-entity (10.31) | dvx | 0.882 | 1.8564 | 0.920 | 0.961 |
| multi-entity (10.31) | dvy | 0.950 | 5.9890 | 0.971 | 0.970 |
| sprint-targeted (10.45) | dx | 0.930 | 0.0000 | 0.930 | 0.796 |
| sprint-targeted (10.45) | dy | 0.895 | 4.4335 | 0.922 | 0.908 |
| sprint-targeted (10.45) | dvx | 0.981 | 2.1119 | 0.996 | 0.989 |
| sprint-targeted (10.45) | dvy | 0.879 | 5.9758 | 0.948 | 0.965 |
| jump-targeted (10.52) | dx | 0.957 | 0.0000 | 0.957 | 0.964 |
| jump-targeted (10.52) | dy | 0.884 | 4.0462 | 0.886 | 0.909 |
| jump-targeted (10.52) | dvx | 0.954 | 2.0771 | 0.806 | 0.993 |
| jump-targeted (10.52) | dvy | 0.924 | 5.2948 | 0.949 | 0.951 |

1. **The guarantee holds where the splits are exchangeable, and the dips name where they are not.** The law arm covers at or above target on 15 of 20 bounded cells, minimum coverage 0.876. The five below-target cells are all dy/dvx/dvy - never dx - on the multi-entity, sprint and jump recordings, whose excitation policies make validation and test episodes differ more than the published walk does. That is the honest reading of what the guarantee is: marginal under exchangeability, so a calibration split that under-represents the test episodes' behaviour dips by one to three points, not by the catastrophic margins the Gaussian arm shows on the same rows.
2. **The Gaussian arm is below its own 90% claim on 3 of 20 cells, minimum 0.806** - its worst cell is the jump-targeted recording's dvx, 0.806 against the law arm's 0.954 on the same rows. It also pays for its tails in width: the median Gaussian half-width is 2.22 times the conformal one. The parametric arm is wider where it does not need to be and still wrong where it matters - the interval-level version of the tail under-prediction 10.58 measured for the residue.
3. **The fit-mean null is the worst arm everywhere**, minimum coverage 0.774 at the same level, so the law's interval is not a re-centred constant band.
4. **The dx cells show 10.59's exactness as a zero-width interval.** On four recordings the 95th percentile of the calibration residual is exactly 0.0000 - the law is exact on more than 95% of calibration frames - so the conformal interval around dx has zero width and still covers 90-96% of test frames. An uncertainty method that can return "this law is exact where it is exact" is the right instrument for an engine whose identity holds on 99.16-99.85% of frames once the repeats are dropped.

**Limitations.** The guarantee is marginal over exchangeable draws, not per-cell: the five below-target cells are the measurement of that boundary, not a violation of the arithmetic. Coverage is scored under the recorded actions, so it is a statement about the transition kernel and not about any planner's state visitation. The recordings still contain the paused frames 10.59 identified - the calibration split scores them like any other frame - and the calibration sizes differ per recording (392 to 3,000 rows), while all 20 cells stayed bounded at every level reported. The law is the robust SINDy fit alone; the least-squares arm of 10.56 is not re-fitted here, and a different estimator needs its own calibration run.

Regenerate: `python -m src.evaluation.conformal_uncertainty_benchmark` (emulator-free, ~10 s, writes `results/conformal_uncertainty_metrics.json`), or `make conformal-uncertainty`.

---
