[← 10.30](10-30-scope-limitations-threats-to-validity.md) · [Contents](../../README.md#contents) · [10.32 →](10-32-hierarchical-global-local-planning-a-cem-mpc.md)

### 10.31 End-to-End Pixel Perception (Pixel-to-Action Front-End)

To close limitation §10.30-2 (privileged WRAM telemetry), the emulator now
captures native RGB frames (opt-in `enable_frame_capture()`, verified
256x224, RGB565/0RGB1555/RGB888 conversion unit-tested) and a CNN
`PixelStateEstimator` (`src/perception/`) regresses frames directly to the 8D
WRAM vector with a fitted `StateNormalizer`. `scripts/record_pixel_gameplay.py`
records paired data; `src/training/train_pixel_estimator.py` trains with
per-variable reports; `src/evaluation/evaluate_pixel_mpc.py` closes the loop
pixels → estimate → Hard-PINN MPC (WRAM read in parallel only to *measure*
estimator error, never for control). Scope is deliberately a supervised
state-estimation front-end, not a pixel-space world model.

#### 10.31.1 Measured Results on Real Hardware

`scripts/record_pixel_gameplay.py` captured **2,720** paired (frame, WRAM)
transitions in **2.8 s** of wall-clock (`data/raw/smw_pixel_dataset.npz`).
The trained estimator (`results/pixel_estimator_metrics.json`, best validation
SmoothL1 **0.1566** on normalized states) reproduces position but not motion:

| Channel | Metric | Value | Reading |
| :--- | :--- | :---: | :--- |
| $X$ | R² / MAE | **0.948** / 22.78 px | scroll position is visible in the background |
| $Y$ | R² / MAE | 0.435 / 14.59 px | altitude partly recoverable from the tile horizon |
| $v_x$ | R² / MAE | 0.116 / 16.46 subpx/f | speed barely identifiable from one frame |
| $v_y$ | R² / MAE | **-0.005** / 23.53 subpx/f | **no better than predicting the mean** |
| `c_ground` | accuracy / F1 | 0.877 / 0.904 | contact is inferable from pose |
| `c_left` / `c_right` / `c_ceiling` | accuracy | 0.993 / 1.000 / 1.000 | degenerate classes (rarely set), F1 undefined or 0 |

The negative result is the point: **velocity is not a function of a single RGB
frame**, so a pixel-to-WRAM regressor cannot feed a dynamics model that consumes
$(v_x, v_y)$ - it needs multi-frame input or an explicit observer. Closed loop
(`results/pixel_mpc_metrics.json`) confirms the cost: pixels → estimator → Hard-PINN
MPC advances **108.94 px in 202 frames** before falling in a pit, against **571.75 px
in 300 frames** for the same planner driven by WRAM truth (§10.6), with a mean
estimation error of 84.27 units over $(x, y, v_x, v_y)$.

**One limit on that sentence, found later.** It was measured on the frames §10.31.1 turns out to be
describing - a composite with nothing moving in it - so it could not have come out the other way.
§10.65 re-runs the same protocol on the scrolling recording of §10.62 and in a 2x2 against it: the
strong form does not survive there, and the multi-frame half of the recommendation gets a number.

**An audit of the frames themselves, added later.** Whether the imagery is the level Mario is walking through is testable without a model: align each consecutive pair by the integer horizontal shift that best matches them, and compare that shift with the change in the recorded x. The alignment does see a scroll when there is one - sliding a frame by two pixels is recovered as two pixels on 100.00% of 136 synthetic trials - and on this file it finds nothing: the best shift is zero on 100.00% of the 2,719 consecutive pairs, the picture moves 0.00 px per frame while the state moves 3.94 px, and the correlation between the two is undefined, because the picture never shifts at all. The frames are therefore not the scrolling level: the savestate this repository records from restores into engine mode `0x08` and the harness writes `0x14` to force interactive physics (§10.38.1), so the WRAM player is simulated while the PPU keeps drawing the title and file-select composite - which is what the exported frames show. The estimator numbers above are what that model achieved on that imagery and they stand as measurements of it; what is withdrawn is the reading beside the first row, position recoverable from the scroll of the background, because there is no background scroll in these frames to recover. `python -m src.evaluation.pixel_frame_probe` re-derives this paragraph from `results/pixel_frame_probe_metrics.json`. §10.62 later re-makes this recording from the level-entry savestate of §10.36.3 and shows the same alignment reporting movement when the camera scrolls and stillness when it does not: the null was the imagery, not the instrument.
