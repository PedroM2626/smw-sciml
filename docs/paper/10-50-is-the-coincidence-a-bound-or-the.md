[← 10.49](10-49-which-documented-constant-does-the-telemetry-actually.md) · [Contents](../../README.md#contents) · [10.51 →](10-51-the-excluded-cell-bounds-projected-onto-a.md)

### 10.50 Is the Coincidence a Bound or the Support?

Section 10.46 published a number it explicitly refused to interpret: the plain DeepONet's implied plateau, 35.61 sub-pixels/frame, sits 1.3% from the sustained sprint ceiling the console was measured at in 10.45 (36.075), while the other five models probed there sat between 21 and 30. The section said the agreement "is not evidence of anything... but it is the kind of number a future study should test rather than smooth over". This is that test.

The instrument is a manipulation rather than an estimator: keep the console and the architecture, change only the support of what the model is allowed to see. Transitions whose next-state velocity exceeds a cap - 49 (the recording as is), 36, 30, 24 - are dropped from the training split, and each resulting model is probed with the 10.46 fixed-point instrument over the *full* recorded range, so the probe cannot be the thing that moved. Three seeds per cell.

| family | training cap | support the model saw | implied plateau | std over seeds | max driven accel. | seeds with a plateau |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepONet | 49.0 | 49.0 | 32.12 | 9.08 | 16.49 | 3 / 3 |
| DeepONet | 36.0 | 36.0 | 28.75 | 11.81 | 17.07 | 3 / 3 |
| DeepONet | 30.0 | 30.0 | 18.18 | 1.80 | 17.86 | 3 / 3 |
| DeepONet | 24.0 | 24.0 | 17.36 | 0.75 | 16.96 | 3 / 3 |
| *DeepONet, plateau vs support* | | slope 0.644 | correlation 0.924 | | | |
| FNO | 49.0 | 49.0 | 23.85 | 3.07 | 2.02 | 2 / 3 |
| FNO | 36.0 | 36.0 | 23.56 | 1.26 | 0.75 | 3 / 3 |
| FNO | 30.0 | 30.0 | 7.00 | 1.65 | 0.33 | 2 / 3 |
| FNO | 24.0 | 24.0 | 8.66 | 2.90 | 0.53 | 2 / 3 |
| *FNO, plateau vs support* | | slope 0.709 | correlation 0.824 | | | |
| MLP | 49.0 | 49.0 | 33.12 | 4.13 | 12.51 | 3 / 3 |
| MLP | 36.0 | 36.0 | 36.31 | 7.44 | 11.76 | 3 / 3 |
| MLP | 30.0 | 30.0 | 39.14 | 2.84 | 8.26 | 3 / 3 |
| MLP | 24.0 | 24.0 | 20.34 | 1.82 | 6.76 | 3 / 3 |
| *MLP, plateau vs support* | | slope 0.329 | correlation 0.424 | | | |

1. **The coincidence was the data, and the number was one draw.** With the recording's fast frames removed, the DeepONet's plateau follows them down: 32.12 with the full support, 17.36 at cap 24, a slope of 0.644 on the support cap with correlation 0.924. The published 35.61 is inside the seed spread of its own untruncated cell ($32.12 \pm 9.08$ over three seeds) - so the 1.3% agreement with 36.075 is neither a property of the operator representation nor a measurement of the console: it is one sample from a distribution whose mean is set by what the model was shown. Section 10.46's decision to record rather than interpret is vindicated; the interpretation is now closed, and it is negative.
2. **The FNO tracks the support too, and more steeply** (0.709, $r = 0.824$, from 23.85 down to 8.66), which is what the 10.46 traction test would have predicted: its plateau at the full support was already one of the "plausible" ones, and it collapses when the data stops telling it anything about the top of the range.
3. **The MLP is the counterexample, and it is instructive.** Its plateau does not track the cap (0.329, $r = 0.424$) and is not even monotone: truncating the support to 30 moves the implied plateau *up* to 39.14, above the fastest frame the model ever saw. A fixed point of the driven map is a statement about where the learned extrapolation folds back, not about where the data stops, and this is the same logic that made 10.43 refuse to report "no fixed point" as the data maximum - now demonstrated from the other side.
4. **Two of the three families plateau below their support, all three plateau far below the run cap that 10.49 establishes.** The untruncated plateaus are 32.12, 23.85 and 33.12 against a documented run bound of 48.0 sub-pixels/frame: on the correct reference the gap is 31% to 50% rather than the 50% to 70% 10.46 computed against the P-meter constant.

**Limitations.** Three seeds per cell is enough to say whether a plateau is stable and not enough to fit a slope with confidence intervals; the slope and correlation are reported as descriptive quantities, which is exactly the reading finding 1 needs (the published value lies inside the observed spread). Truncating the training split changes the class balance of the *upper* velocity band, so an arm trained at cap 24 also sees proportionally fewer sustained-run frames; the probe scan range is held at the full recorded support in every cell to keep the comparison honest, and where no fixed point exists the cell records "none" rather than the data maximum, following 10.43's rule.

![Does the learned plateau follow the bound or the support?](../../results/figures/plateau_provenance.png)

*Implied plateau against the velocity cap applied to the training split, three seeds per point. A plateau that were the engine's bound would stay flat as the cap is lowered; the dotted diagonal is what a pure support artifact looks like. The DeepONet and the FNO fall with the support; the MLP does not move monotonically at all.*

Regenerate: `python -m src.evaluation.plateau_provenance_benchmark` (emulator-free, ~15 min for 3 families × 4 caps × 3 seeds; `make plateau-provenance`; smoke config `configs/smoke_plateau_provenance.yaml`; `--caps`, `--seeds` and `--epochs` control the budget).
