[← 10.74](10-74-aquatic-stage-clearance-by-physics-informed-reinforcement-learning.md) · [Contents](../../README.md#contents) · [11 →](11-complete-reproducibility-guide.md)

### 10.75 What the Physics-Informed Terms of PIML-PPO Contribute: a Four-Arm Ablation

10.71 and 10.74 report closed-loop clears by `PIML-PPO` and describe them as physics-informed reinforcement learning, and 10.71.3 and 10.74.5 record two things the clears do not show: every run trains with a demonstration anchor (a cross-entropy term on the reference trajectory in every PPO update, then a polish until the network agrees with every anchor action), and no arm trained without the physics-informed terms exists. `src/evaluation/piml_sciml_ablation.py` trains four arms on the same stages, seeds and budget and evaluates each in deterministic closed loop from the stage's first frame.

| arm | physics-informed terms (CBF logit filter, HJB critic term, action-physics term, Hard PINN head) | self-imitation and polish |
|---|---|---|
| `full` | on, as shipped | on |
| `no_sciml` | off (`cbf_beta = lambda_hjb = lambda_phys_act = lambda_pinn = 0`) | on |
| `no_imitation` | on | off (`lambda_sil = 0`, no polish) |
| `no_imitation_no_sciml` | off | off |

Two things stay in every arm because they are the demonstration entering by another door: the frontier resets (rollouts start from savestates along the reference trajectory) and the value anchor (the critic is fitted to the reference trajectory's returns). Each cell is 3 seeds (42, 43, 44) of 25 PPO epochs on the GPU with `TF32` off, evaluated twice: right after the PPO epochs, and after the polish. The shipped checkpoints of `0x29` and `0x27` used 35 epochs; `0x26` used 25.

| stage | arm | clears after PPO | mean progress after PPO | clears after polish | mean progress after polish | mean polish steps |
|---|---|---|---|---|---|---|
| `0x29` | `full` | 0 / 3 | 1738.4 px | 1 / 3 | 3547.6 px | 107 |
| `0x29` | `no_sciml` | 0 / 3 | 2524.3 px | 2 / 3 | 4473.1 px | 119 |
| `0x29` | `no_imitation` | 0 / 3 | 197.5 px | 0 / 3 | 197.5 px | 0 |
| `0x29` | `no_imitation_no_sciml` | 0 / 3 | 548.1 px | 0 / 3 | 548.1 px | 0 |
| `0x27` | `full` | 0 / 3 | 365.5 px | 3 / 3 | 5065.6 px | 366 |
| `0x27` | `no_sciml` | 0 / 3 | 338.4 px | 0 / 3 | 2813.2 px | 344 |
| `0x27` | `no_imitation` | 0 / 3 | 236.9 px | 0 / 3 | 236.9 px | 0 |
| `0x27` | `no_imitation_no_sciml` | 0 / 3 | 200.5 px | 0 / 3 | 200.5 px | 0 |
| `0x26` | `full` | 1 / 3 | 1905.7 px | 3 / 3 | 2735.6 px | 337 |
| `0x26` | `no_sciml` | 1 / 3 | 2260.4 px | 3 / 3 | 2735.6 px | 283 |
| `0x26` | `no_imitation` | 0 / 3 | 859.7 px | 0 / 3 | 859.7 px | 0 |
| `0x26` | `no_imitation_no_sciml` | 0 / 3 | 1411.6 px | 0 / 3 | 1411.6 px | 0 |

**Without the demonstration anchor nothing clears.** The 18 `no_imitation` and `no_imitation_no_sciml` runs clear no stage: 0 of 9 for each arm, and the furthest any of them gets is 1830 px (`0x26`, `no_imitation_no_sciml`), where the clears reach 2736 to 5066 px of progress. PPO with frontier resets and the value anchor does not find the way across `0x29`, `0x27` or `0x26` on this budget, with or without the physics-informed terms.

**With the anchor, the clear is made by the polish.** Of the 18 `full` and `no_sciml` runs, 2 clear right after the PPO epochs (both `0x26`, seed 44) and 12 after the polish. After the PPO epochs the network agrees with 95% to 99.8% of the anchor actions and still dies in closed loop - `0x29` seed 42 reaches 2144 px - until between 68 and 444 cross-entropy steps make it agree with every one.

**The physics-informed terms: the data cannot say.** With the anchor, `full` clears 7 of 9 runs and `no_sciml` 5 of 9 (Fisher exact test, p = 0.62), and the sign changes with the stage: on `0x29` 1 of 3 against 2 of 3, on `0x27` 3 of 3 against 0 of 3 (p = 0.10), on `0x26` 3 of 3 against 3 of 3. Three seeds cannot tell a contribution from noise, and the one stage where the difference is largest, `0x27`, is a hypothesis for a larger study, not a result. What the table does rule out is the reading the earlier sections left open: the clears are not something physics-informed reinforcement learning achieves on its own.

**One seed is not a clear rate.** The `full` arm on `0x29` clears with seed 42 - the shipped result, `4808.0625 px` - and reaches `3139 px` and `2695 px` with seeds 43 and 44. The "100% closed-loop completion" of 10.71.3 is one successful draw of a procedure that succeeds in 1 of 3.

**What this does not show.** That the physics-informed terms matter in a setting without a demonstration, or on a larger budget; that `0x27`'s 3 of 3 against 0 of 3 is more than chance; and anything about seeds beyond the three used. The demonstration also enters through the frontier resets and the value anchor in every arm, so the `no_imitation` arms are "no cloning", not "no demonstration".

Regenerate: `make piml-sciml-ablation` (or `smw-pinn piml-sciml-ablation`; needs core + ROM, about an hour on a GPU). Artifact: `results/piml_sciml_ablation_metrics.json`.
