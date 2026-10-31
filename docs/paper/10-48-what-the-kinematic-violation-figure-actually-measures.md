[← 10.47](10-47-is-it-the-shell-or-the-parameterisation.md) · [Contents](../../README.md#contents) · [10.49 →](10-49-which-documented-constant-does-the-telemetry-actually.md)

### 10.48 What the "Kinematic Violation" Figure Actually Measures

Every physics claim in Sections 8, 10.27 and 10.42 is carried by one number: the rollout *kinematic-violation rate*, computed by `RolloutEvaluator` as a frame where `|Δx − v/16| > 0.2 px` - with `v` the velocity the model reported on the **previous** frame. Section 10.47 met the consequence: an arm that integrates position exactly by construction was still flagged on 16.18% of its frames. `src/evaluation/rollout_diagnostics.py` separates the properties that number mixes and re-scores every committed model - the six of Section 8/10.41/10.42 and the fifteen of 10.47 - with each one reported on its own, plus the recorded telemetry itself as the reference row.

| model | shell in graph | published violation | integration residual (median px) | velocity-jump rate | bound-exceedance rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| mlp | no | 0.9790 | 2.05e+00 | 0.0092 | 0.4383 |
| soft_pinn | no | 1.0000 | 1.34e+00 | 0.0000 | 0.0000 |
| hard_pinn | yes | 0.0000 | 6.50e-06 | 0.0000 | 0.0000 |
| deeponet | no | 0.9958 | 1.96e+00 | 0.0109 | 0.0000 |
| physics_constrained_deeponet | yes | 0.0000 | 6.94e-06 | 0.0000 | 0.0000 |
| fno | no | 0.8849 | 8.59e-01 | 0.1109 | 0.3275 |
| grid:MLP/state/none | no | 0.9790 | 2.05e+00 | 0.0092 | 0.4383 |
| grid:MLP/state/soft | no | 1.0000 | 1.33e+00 | 0.0000 | 0.0000 |
| grid:MLP/residual/none | no | 0.0000 | 6.51e-06 | 0.0000 | 0.4658 |
| grid:MLP/residual/soft | no | 0.0000 | 7.15e-06 | 0.0000 | 0.0000 |
| grid:MLP/residual/hard | yes | 0.0000 | 6.29e-06 | 0.0000 | 0.0000 |
| grid:DeepONet/state/none | no | 0.9958 | 1.96e+00 | 0.0109 | 0.0000 |
| grid:DeepONet/state/soft | no | 1.0000 | 3.80e+00 | 0.0294 | 0.0025 |
| grid:DeepONet/residual/none | no | 0.0000 | 6.48e-06 | 0.0000 | 0.2700 |
| grid:DeepONet/residual/soft | no | 0.0000 | 6.61e-06 | 0.0000 | 0.0000 |
| grid:DeepONet/residual/hard | yes | 0.0000 | 6.94e-06 | 0.0000 | 0.0000 |
| grid:FNO/state/none | no | 0.8849 | 8.59e-01 | 0.1109 | 0.3275 |
| grid:FNO/state/soft | no | 0.8773 | 1.48e+00 | 0.0025 | 0.0000 |
| grid:FNO/residual/none | no | 0.0126 | 5.90e-06 | 0.0126 | 0.4050 |
| grid:FNO/residual/soft | no | 0.1975 | 9.20e-06 | 0.1975 | 0.4150 |
| grid:FNO/residual/hard | yes | 0.0000 | 6.83e-06 | 0.0000 | 0.0000 |
| *recorded telemetry (the console itself)* | - | 0.0462 | 5.63e-02 | 0.0303 | 0.1183 |

1. **For an exact integrator the published figure is not a consistency measurement at all - it is the jump rate, digit for digit.** The eleven models whose integration residual is below $10^{-4}$ px include `grid:FNO/residual/none` (residual $5.90\times10^{-6}$ px, violation 0.0126, jump rate 0.0126) and `grid:FNO/residual/soft` (residual $9.20\times10^{-6}$ px, violation 0.1975, jump rate 0.1975). The two columns are equal because that is what the predicate computes once the position identity holds: it fires on a velocity change of more than 3.2 sub-pixels/frame. A "0% violation rate" therefore means *the model never accelerated by more than 3.2 sub-pixels in a frame*, which for the shells is a property of what they learned, not of what the graph guarantees.
2. **The console's own telemetry is flagged on 4.62% of frames, so the scale runs from "as consistent as Mario", not from "zero".** Recorded transitions jump on 3.03% of frames and carry a median integration residual of 0.0563 px against the velocity they advanced by. A model at 0.0462 is exactly as self-consistent as the engine; the repository has been reading 0.0000 as the only acceptable value when it is in fact smoother than reality.
3. **Zero violations is not a statement about the velocity bounds either.** `grid:MLP/residual/none` and `grid:DeepONet/residual/none` both post 0.0000 on the published metric while their rollouts leave the engine's bounds on 46.58% and 27.00% of frames. This is the same dissociation 10.47 measured for the shells' drift, now visible inside the metric the older tables published: consistency, smoothness and boundedness are three numbers, and the repository has been quoting one of them as if it covered all three.
4. **The traction threshold that 10.46 chose is load-bearing, and the re-classification it invited changes the answer.** Re-scoring every probed ceiling of 10.46 and 10.47 under four thresholds gives accepted sets of 3, 3, 9 and 9 records at 1.0, 1.5, 2.5 and 3.5 px/frame. The three 10.46 acceptances - FNO at 1.61, Hard PINN at 1.72, PC-DeepONet at 1.91 - only appear at the published 2.5 or looser; at 1.5 the six published models contribute *none*, and the three survivors are all 10.47 grid arms (0.08-0.58 px/frame). 10.46's headline should therefore be read as "three of six pass a traction test set at 2.5 px/frame", which is what its own table says, and not as a threshold-free finding.

**Limitations.** The jump-rate equivalence of finding 1 is exact only because the predicate's tolerance is the only active term once $\Delta x = \hat v_{t+1}/16$; a model that integrates with the *previous* velocity would couple the two columns differently, and 10.49 shows the console actually integrates that way. Rollouts are open-loop on the recorded action sequences, so a jump rate partly reflects the recorded policy's own acceleration. The reference row uses `next_states` as the trajectory, which inherits the console's real screen wraps: the integration column is reported as a median precisely because a wrap is a tail event of hundreds of pixels, not a physical one.

Regenerate: `python -m src.evaluation.kinematic_metric_decomposition_benchmark` (emulator-free, ~1 min; `make metric-decomposition`; `--num-starts`, `--horizon` and `--config configs/smoke_metric_decomposition.yaml` control the budget).
