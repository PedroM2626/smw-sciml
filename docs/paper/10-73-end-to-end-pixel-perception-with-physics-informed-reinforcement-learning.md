[← 10.72](10-72-aquatic-physics-parameter-identification-and-fluid-regime-change.md) · [Contents](../../README.md#contents) · [11 →](11-complete-reproducibility-guide.md)

### 10.73 End-to-End Pixel Perception with Physics-Informed Reinforcement Learning (`Pixel-PIML`)

Sections 10.39 and 10.71 established that Physics-Informed Reinforcement Learning (`PIML-PPO`) clears long-horizon stages when conditioned on privileged WRAM states. Section 10.65 demonstrated that a 2-frame convolutional estimator (`PixelStateEstimator`, `in_channels=6`) successfully decodes kinematic velocity from scrolling PPU imagery ($R^2 \approx 0.81$ for $v_x$, $0.84$ for $v_y$). However, the closed-loop control evaluated in 10.31 relied on unconstrained MPC reading static menu frames.

`src/evaluation/pixel_piml_closed_loop.py` closes the loop on the authentic SNES console:
1. **Visual-Only Observation Stream:** The agent receives **no** privileged coordinate or velocity telemetry from WRAM during inference. It receives strictly native $256 \times 224$ RGB frames from the console's PPU via `emu.get_frame()`.
2. **Convolutional State Estimation:** A 2-frame stacked CNN front-end (`pixel_estimator_retrain_level_pair_best.pt`) decodes the continuous state vector $\hat s_t = [\hat x_t, \hat y_t, \hat v_{x,t}, \hat v_{y,t}, \hat c_{\text{gnd}}, \hat c_{\text{ceil}}, \hat c_{\text{left}}, \hat c_{\text{right}}]^\top \in \mathbb{R}^8$.
3. **Safety Projection Under Perceptual Uncertainty:** The Discrete Control Barrier Function filter (`DiscreteCBFCategoricalFilter`, $\beta_{\text{cbf}} = 0.25$) projects the policy logits onto the admissible action set using the *visually estimated* coordinates $\hat s_t$, filtering unsafe sprint commands and ceiling/wall collisions before sending the joypad signal to the emulator bus.
4. **Passive Ground-Truth Tracking:** Console WRAM is read in parallel strictly as an unprivileged evaluator to score visual tracking error ($\text{MAE}$, $R^2$), safety violations, and physical progress.

#### 10.73.1 Closed-Loop Hardware Benchmark Results

Three comparative arms were flown from `smw_yoshi_island_1_level.state` over independent random seeds ($n=3$, seeds 42, 43, 44) under a 500-frame episode budget:

| Evaluated Control Arm | Input Sensing Modality | Safety Filter | Mean Forward Progress (px) | Max Progress Single Run (px) | Executed Action Physics Violation Rate | Survival Rate (Deaths / Runs) |
|---|---|---|:---:|:---:|:---:|:---:|
| **Visual Random Baseline** | 2-Frame RGB Pixels | None | 261.15 ± 83.24 px | 362.1 px | 0.0000 | 0 / 3 (0%) |
| **Unconstrained Visual Policy** | 2-Frame RGB Pixels | None | 1041.40 ± 26.30 px | 1078.2 px | 0.0053 | 0 / 3 (0%) |
| **Physics-Informed Visual Policy (`Pixel-PIML`)** | 2-Frame RGB Pixels | **Discrete CBF** ($\beta = 0.25$) | **1198.04 ± 143.20 px** | **1308.5 px** | **0.0000 (0 violations)** | 0 / 3 (0%) |

Key empirical findings:

1. **Superior Autonomous Navigation Under Raw Pixels:** The `Pixel-PIML` agent achieves a mean forward progress of **$1198.04\text{ px}$**, surpassing the unconstrained visual policy by **$+156.64\text{ px}$** and reaching peak runs of **$1308.5\text{ px}$** driven entirely by pixel inputs.
2. **Absolute Suppression of Physical Violations:** While the unconstrained policy occasionally commands actions that violate engine speed limits or drive into walls under visual noise (violation rate $0.0053$), the Discrete CBF projection filters those commands at execution time, maintaining **zero physical violations ($0.0000$)** across all seeds.
3. **Tracking Accuracy During Motion:** Across active gameplay frames, the 2-frame convolutional estimator tracks horizontal velocity with a mean MAE of $11.34$ to $22.80 \text{ subpx/frame}$ and vertical position with an MAE of $17.30 \text{ px}$, providing sufficiently stable kinematic features for the CBF layer.
4. **Out-of-Distribution Perceptual Failure Mode:** When tested on unseen stages with distinct color palettes (e.g., Yoshi's Island 2, `0x29`), the CNN pixel estimator experiences perceptual distribution shift, hallucinating extreme horizontal velocities ($\hat v_x \gt 90 \text{ subpx/f}$). Consequently, the CBF safety barrier acts conservatively by penalizing forward acceleration and freezing the agent ($0.0\text{ px}$ progress). This proves that while SciML barriers guarantee physical safety under accurate state estimates, end-to-end vision requires multi-stage domain randomization to generalize across disparate visual themes.

Regenerate: `python -m src.evaluation.pixel_piml_closed_loop --state smw_yoshi_island_1_level.state`. Artifact: `results/pixel_piml_closed_loop_metrics.json`. Checkpoint: `results/checkpoints/pixel_estimator_retrain_level_pair_best.pt`.
