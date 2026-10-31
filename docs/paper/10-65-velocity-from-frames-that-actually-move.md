[← 10.64](10-64-the-buffer-the-telemetry-reads-is-the.md) · [Contents](../../README.md#contents) · [10.66 →](10-66-the-same-dashboard-on-every-stage-the.md)

### 10.65 Velocity From Frames That Actually Move

10.31 concluded that velocity is not a function of a single RGB frame, and 10.31.1 discovered the imagery behind that conclusion was the title and file-select composite - frames with nothing moving in them. The claim was measured on data that could not have supported the opposite answer, so it is re-measured here as a 2x2: the published recording and the scrolling one of 10.62, each fed one frame and each fed a frame stacked against its predecessor. The architecture, seed, split and metric are the published ones; only the first convolution widens, and only for the two-frame cells.

| cell | recording | input | frames | $R^2$ of $v_x$ | $R^2$ of $v_y$ | $R^2$ of $x$ | MAE of $v_x$ (subpx/f) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `published_single` | smw_pixel_dataset.npz | single | 2,720 | 11.60% | -0.52% | 94.82% | 16.4596 |
| `published_pair` | smw_pixel_dataset.npz | pair | 2,720 | 29.64% | 0.20% | 93.36% | 13.7051 |
| `level_single` | smw_pixel_level_dataset.npz | single | 4,112 | 52.68% | 45.33% | 97.79% | 10.8960 |
| `level_pair` | smw_pixel_level_dataset.npz | pair | 4,112 | 81.04% | 83.97% | 99.11% | 6.6810 |

Two things follow, and they pull in opposite directions. Down the single-frame column, $v_x$ goes from $R^2$ 11.60% on the menu frames to 52.68% on the level and $v_y$ from -0.52% to 45.33%: a single frame of imagery that actually moves does carry speed, so the strong form of 10.31's sentence - that it cannot - does not survive. 10.31's sentence was measured on frames that could not show motion, and the correction belongs in its limitations note, which is where this section puts it. Across the scrolling rows the recommendation is confirmed and quantified: stacking the predecessor buys +0.2836 $R^2$ on $v_x$ and +0.3864 on $v_y$, to 81.04% and 83.97% - the first time this repository has measured that gain rather than asked for it, since the old recording had no motion for a second frame to reveal.

The two controls are what make those readings safe. The first cell is 10.31's own run read from its own artifact, so the protocol was carried over rather than re-invented; and the second cell shows the pair is not simply better - on the menu frames it degrades pose while lifting $v_x$ to 29.64% from 11.60%, which is the warning that a wider input is not a free win. Position is the row to read first: $R^2$ of $x$ was already 94.82% on frames that show nothing, which is the split's optimism and not a model's achievement - see the limitations.

**Limitations.** The split is random over frame indices, as the published protocol is, and a 60 Hz recording puts near-duplicates on both sides of such a split; every cell shares that, including the parity cell, so the comparisons are fair and the absolute numbers are upper bounds - the $x$ row above is the clearest symptom. Pairs are stacked in a fixed order and a recording's first frame is paired with itself. One seed per cell, and the largest gap in the table is 0.4585 $R^2$, so the ordering is a single draw rather than a distribution. The published checkpoint and its artifact are read, never rewritten; the new cells are `pixel_estimator_retrain_*` files.
