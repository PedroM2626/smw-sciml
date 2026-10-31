# Physics-Informed Neural Networks (PINN) vs. Statistical Models in Super Mario World
## Discrete Dynamics Modeling Without Computer Vision: An Empirical Benchmark on Sample Efficiency and Inductive Physics Biases

**Author:** Pedro Morato Lahoz  
**Hardware Acceleration:** NVIDIA GeForce RTX 4070 Laptop GPU (PyTorch 2.5.1 + CUDA 12.1)  
**Execution Environment:** Headless Libretro Ctypes Emulation (Snes9x Core v1.63, 60 FPS, 128 KB WRAM)  
**Base ROM:** *Super Mario World (USA)* — SHA-1: `6B47BB75D16514B6A476AA0C73A683A2A4C18765`  
**Dataset:** 8,077 genuine frame-by-frame transitions (60 Hz) recorded in Interactive Gameplay Mode `$7E:0100 = 0x14`; §10.64 identifies the tile buffer they were recorded against as the title screen's own, not a stage the harness entered  

---

## What the Actor Sees, and What It Believed

![One clearance-draw episode on the live console: the emulator's own RGB picture filling the left, and on the right the tile window the level-conditioned models read, what the actor thinks as eight named actions, the buttons the console actually received, the terminal-value critic's prediction, and along the bottom the twelve numbers the actor sees and the stage progress bar](results/figures/dashboard_run.gif)

*One episode of the `mpc` controller, recorded frame by frame from the committed level-entry savestate `smw_yoshi_island_1_level.state`: 6706 frames, 5010.0000 px of forward progress, ending as `engine_left_0x14_to_0x0C`. The engine mode byte reads `0x0C`, `0x0D`, `0x0E`, `0x14` on the frames the player was live; the mean top button probability is 0.9460; the section-4.1 identity holds exactly on 73.75% of the live frames and the exceptions are the 2 intervals where the engine holds the player still, which 10.60 identifies as the level's own message box. Full quality: [`results/figures/dashboard_run.mp4`](results/figures/dashboard_run.mp4) (4.28 MB, 9.30 s) - a download, not a player: GitHub serves a committed `.mp4` as a file, which is why the inline copy above is a GIF. 10.60 says which controller flies this, and why. It is one of 4 stages - `0x27`, `0x28`, `0x29`, `0x2A`, 1 of which the clearance controller has now finished on the console's own terms - that the world map actually hands over, and 10.66 shows the rest and says what a clear takes.*

## 🏆 Direct Answer: Which Model Performed Best in the Benchmark?

Within **the four architectures of the canonical Section 8 benchmark**, the **Hard Residual PINN (Hard Physics Constraints / Structural Inductive Bias)** achieved the lowest reported prediction error and the strongest kinematic consistency. That scope is load-bearing and the repository now measures two things outside it: Section 10.42 fits two operators on the same split and seed and both reach a lower single-step error (`FNO` 0.4025 and `PhysicsConstrained_DeepONet` 0.5766 against 0.5783), and Section 10.37's hand-written engine rules carry an **identically zero** kinematic residual with **no learned parameters at all**. Section 10.53 narrows the consistency claim too: the 0.0% violation rate below is a result *inside the published 0.2 px tolerance*, and at 0.002 px the same shells are flagged on 0.9345-0.9578 of their rollout frames while a model that integrates position the way the console does stays at exactly zero.

Sections 10.37-10.69 train and score the rest of the repository's models against the numbers below - the hand-written engine rules, the parametrically identified map, the discovered symbolic laws, the operator family (DeepONet, Physics-Constrained DeepONet, FNO), the physics-injection grid, the effective-velocity grid, the neural-ODE solver grid, the SINDy identifications, the corrected-physics ablation, the residue measurement of section 4.1, the camera channel that measurement predicted, the dashboard at the top of this file that says which actor flies it, the terrain comparison that says which level the recordings walk, the pixel re-recording that turns that reading into a measured discrimination, the held-player split that says which mechanism the recorder has been sampling, the buffer identification that says whose terrain the tile-conditioned results are conditioned on, the retraining that re-measures the pixel conclusion on frames that move, and the world-map survey that says how many stages there are to be conditioned on at all, re-run in 10.67 to check that it answers the same way twice - and every number the survey and its successors published about what those stages hold is re-measured off the engine itself in 10.68, which wrote each tile id into the buffer and read the console's own answer back - so this table is the canonical comparison, not the set of models this repository has measured.

### Key Factors in the Performance of the Hard Residual PINN
1. **Single-Step Predictive Accuracy (Test MSE) - the four Section 8 architectures:**
   * **Hard Residual PINN:** **0.5783**
   * **Statistical MLP:** **16.4717** (**28.5x higher error**)
   * **Soft-Constrained PINN:** **53.8167** (**93.1x higher error**)
   * **Statistical LSTM:** **39.2194** (**67.8x higher error**)
2. **Kinematic Consistency and Physical Constraint Adherence:**
   * The analytical kinematic residual ($\|\Delta X - v_x/16.0\|^2$) of the Hard PINN was **0.0019** (analytical zero within float32 numerical precision limits), compared to **17,561.24** for the MLP and **37,361.16** for the LSTM.
   * In multi-step autoregressive rollouts (120 frames / 2 seconds), the Hard PINN strictly adhered to the discrete kinematic position update constraint (**0 violations across 120 frames, or 0.0%**, under the published predicate's 0.2 px tolerance - 10.48 splits what that figure mixes, and 10.53 shows the zero is tolerance-bounded rather than exact). In contrast, unconstrained statistical baselines exhibited departures from the discrete kinematic update relation across evaluated rollout frames.
3. **High Sample Efficiency (>25x):**
   * Trained with only **$N = 200$ real transitions** (~3.3 seconds of gameplay), the Hard PINN achieved a Test MSE of **0.6764** and an open-loop rollout drift of **41.89 px**.
   * The Statistical MLP required over **$N = 5,000$ transitions** (~83 seconds of gameplay) to reach a Test MSE of **12.4001** and a drift of **85.90 px**.
   * Within the evaluated dataset range, the Hard PINN trained on 200 samples yielded lower prediction error than the MLP trained on 5,000 samples, reflecting a sample efficiency advantage exceeding a factor of **25**.
4. **Parameter and Computational Compactness:**
   * The Hard PINN needs **36,486 parameters** - 0.7% fewer than the black-box MLP (36,744) and 83.7% fewer than the LSTM (223,368) - so its advantage is not bought with capacity, and it converges with high numerical stability within 5 training epochs.

---

## Contents

The full research document lives in [`docs/paper/`](docs/paper/), one page per section, so
GitHub renders every formula on every page (a single page stops rendering math after roughly
1,368 expressions - Section 12 measured it on the README this list replaces). The same pages
are published with MathJax on the [documentation site](https://pedrom2626.github.io/smw-sciml/).

1. [Project Overview & Abstract](docs/paper/01-project-overview-abstract.md)
2. [Introduction and Scientific Motivation](docs/paper/02-introduction-and-scientific-motivation.md)
3. [SNES Hardware Architecture & WRAM Reverse Engineering](docs/paper/03-snes-hardware-architecture-wram-reverse-engineering.md)
4. [Mathematical Formulation of Super Mario World Physics](docs/paper/04-mathematical-formulation-of-super-mario-world-physics.md)
5. [Evaluated Machine Learning Architectures](docs/paper/05-evaluated-machine-learning-architectures.md)
6. [Loss Function Formulation and the Soft PINN Dilemma](docs/paper/06-loss-function-formulation-and-the-soft-pinn.md)
7. [Experimental Protocol, Data Partitioning & Hyperparameters](docs/paper/07-experimental-protocol-data-partitioning-hyperparameters.md)
8. [Empirical Results and Comparative Benchmark Tables](docs/paper/08-empirical-results-and-comparative-benchmark-tables.md)
9. [Visual Analysis of Trajectories and Convergence](docs/paper/09-visual-analysis-of-trajectories-and-convergence.md)
10. **In-Depth Academic Discussion & Critical Analysis**
   * [10.1 Quantitative Answer: How Much Does Knowing the Physics Help?](docs/paper/10-01-quantitative-answer-how-much-does-knowing-the.md)
   * [10.2 The Soft PINN Fallacy in Discrete Dynamical Systems](docs/paper/10-02-the-soft-pinn-fallacy-in-discrete-dynamical.md)
   * [10.3 Analysis of LSTM Performance and the Markovian Hypothesis](docs/paper/10-03-analysis-of-lstm-performance-and-the-markovian.md)
   * [10.4 Long-Horizon Rollout Dispersion (Frame 120 Drift)](docs/paper/10-04-long-horizon-rollout-dispersion-frame-120-drift.md)
   * [10.5 Implications for Model-Based Reinforcement Learning (MBRL)](docs/paper/10-05-implications-for-model-based-reinforcement-learning-mbrl.md)
   * [10.6 Closed-Loop Model-Based RL (MBRL) via Model Predictive Control (MPC)](docs/paper/10-06-closed-loop-model-based-rl-mbrl-via-model-predictive.md)
   * [10.7 Amortized Policy Optimization (Dyna-PPO) & Zero-Shot Model-to-Real Transfer](docs/paper/10-07-amortized-policy-optimization-dyna-ppo-zero-shot-model-to-real-transfer.md)
   * [10.8 Dynamic Hazard Perception (WRAM Sprites & Rex Evasion)](docs/paper/10-08-dynamic-hazard-perception-wram-sprites-rex-evasion.md)
   * [10.9 Canonical Model-Free PPO Baseline vs. PINN-MBRL (Sample Efficiency Triad)](docs/paper/10-09-canonical-model-free-ppo-baseline-vs-pinn-mbrl-sample.md)
   * [10.10 Deep Ensemble of Hard PINNs & Epistemic Uncertainty Quantification](docs/paper/10-10-deep-ensemble-of-hard-pinns-epistemic-uncertainty.md)
   * [10.11 Spatial Translation-Invariant PINN Dynamics](docs/paper/10-11-spatial-translation-invariant-pinn-dynamics.md)
   * [10.12 Closed-Loop Active Model-Based Policy Optimization (Online MBPO)](docs/paper/10-12-closed-loop-active-model-based-policy-optimization-online-mbpo.md)
   * [10.13 Multi-Entity 12D PINN World Model & Autonomous Hazard Evasion](docs/paper/10-13-multi-entity-12d-pinn-world-model-autonomous-hazard.md)
   * [10.14 Safe Model-Based Reinforcement Learning (Deep Ensemble Safe MBPO)](docs/paper/10-14-safe-model-based-reinforcement-learning-deep-ensemble-safe.md)
   * [10.15 Comprehensive Hardware & Computational Efficiency Profiling](docs/paper/10-15-comprehensive-hardware-computational-efficiency-profiling.md)
   * [10.16 Out-of-Distribution (OOD) Zero-Shot Cross-Stage Generalization (Stage A $\to$ Stage B)](docs/paper/10-16-out-of-distribution-ood-zero-shot-cross-stage-generalization-stage-a.md)
   * [10.17 Synchronized Multi-Model Visualization (Real SNES vs. PINN vs. MLP)](docs/paper/10-17-synchronized-multi-model-visualization-real-snes-vs-pinn.md)
   * [10.18 Genuine Multi-Entity Dataset & Supervised Hazard Dynamics Training](docs/paper/10-18-genuine-multi-entity-dataset-supervised-hazard-dynamics-training.md)
   * [10.19 Autonomous Multi-Entity MPC Planning on Real SNES Console (782 px Rex Evasion)](docs/paper/10-19-autonomous-multi-entity-mpc-planning-on-real-snes.md)
   * [10.20 Spatial Discrete Tilemap Perception via WRAM (`$7E:C800`) & Tilemap-PINN](docs/paper/10-20-spatial-discrete-tilemap-perception-via-wram-7e.md)
   * [10.21 Empirical Grounding of Tilemap-PINN (WRAM `$7E:C800` Dataset & 98.66% Contact Accuracy)](docs/paper/10-21-empirical-grounding-of-tilemap-pinn-wram-7e-c800.md)
   * [10.22 Permutation-Invariant Set Multi-Entity World Model (Cross-Attention for N Sprites)](docs/paper/10-22-permutation-invariant-set-multi-entity-world-model-cross-attention-for.md)
   * [10.23 Amortized Policy Distillation from Live MPC Decisions (2,900 FPS vs 23.7 FPS)](docs/paper/10-23-amortized-policy-distillation-from-live-mpc-decisions.md)
   * [10.24 Autonomous Extended Level Navigation on Real SNES Hardware (1,016+ px Progress)](docs/paper/10-24-autonomous-extended-level-navigation-on-real-snes.md)
   * [10.25 Multi-Iteration Interactive DAgger Policy (831 px Progress & 2,707 FPS)](docs/paper/10-25-multi-iteration-interactive-dagger-policy-831-px-progress.md)
   * [10.26 Comprehensive Ablation Study (Clamping, Horizon Drift & CEM Sensitivity)](docs/paper/10-26-comprehensive-ablation-study-clamping-horizon-drift-cem.md)
   * [10.27 Master Algorithm Comparison Table (World Models, MPC & Reactive Policies)](docs/paper/10-27-master-algorithm-comparison-table-world-models-mpc.md)
   * [10.28 Frontier 2: Zero-Shot Closed-Loop Control on Unseen Stage B (*Yoshi's House*)](docs/paper/10-28-frontier-2-zero-shot-closed-loop-control-on-unseen.md)
   * [10.29 Consolidation of Frontiers A, B, C and D: Differentiable Optimization, PPO and Multimodal Rendering](docs/paper/10-29-consolidation-of-frontiers-a-b-c-and.md)
   * [10.30 Scope, Limitations & Threats to Validity](docs/paper/10-30-scope-limitations-threats-to-validity.md)
   * [10.31 End-to-End Pixel Perception (Pixel-to-Action Front-End)](docs/paper/10-31-end-to-end-pixel-perception-pixel-to-action-front-end.md)
   * [10.32 Hierarchical Global + Local Planning (A* + CEM-MPC)](docs/paper/10-32-hierarchical-global-local-planning-a-cem-mpc.md)
   * [10.33 MPC Reflex Ablation (Pure vs Reflexive) & TD-MPC Terminal Value](docs/paper/10-33-mpc-reflex-ablation-pure-vs-reflexive-td-mpc.md)
   * [10.34 Connected Orphans: Tilemap Closed-Loop, Unified Joint Training, Set-12](docs/paper/10-34-connected-orphans-tilemap-closed-loop-unified-joint-training.md)
   * [10.35 Formal Learning Curves & Spatial-Holdout OOD with Danger](docs/paper/10-35-formal-learning-curves-spatial-holdout-ood-with-danger.md)
   * [10.36 Yoshi's Island 2 Capture: Blocked with Full Diagnostics](docs/paper/10-36-yoshi-s-island-2-capture-blocked-with.md)
   * [10.37 Analytical and Oracle-Model Baselines](docs/paper/10-37-analytical-and-oracle-model-baselines.md)
   * [10.38 Closed-Loop Reproduction Audit and Preamble Probe](docs/paper/10-38-closed-loop-reproduction-audit-and-preamble-probe.md)
   * [10.39 Physics-Informed Model-Free RL (PIML-MFRL)](docs/paper/10-39-physics-informed-model-free-rl-piml-mfrl.md)
   * [10.40 Physics Parameter Identification: the Inverse Problem](docs/paper/10-40-physics-parameter-identification-the-inverse-problem.md)
   * [10.41 Neural-Operator Baselines: DeepONet on Discrete Engine Dynamics](docs/paper/10-41-neural-operator-baselines-deeponet-on-discrete-engine-dynamics.md)
   * [10.42 Physics-Constrained DeepONet and the Fourier Neural Operator](docs/paper/10-42-physics-constrained-deeponet-and-the-fourier-neural-operator.md)
   * [10.43 Symbolic Regression as an Inverse-Problem Method: Discovering the Law](docs/paper/10-43-symbolic-regression-as-an-inverse-problem-method-discovering.md)
   * [10.44 Closed-Loop Control with Inverse-Problem World Models](docs/paper/10-44-closed-loop-control-with-inverse-problem-world-models.md)
   * [10.45 Does the Velocity Ceiling Become Measurable Under Targeted Excitation?](docs/paper/10-45-does-the-velocity-ceiling-become-measurable-under.md)
   * [10.46 Do Learned Dynamics Models Contain the Engine's Constraints?](docs/paper/10-46-do-learned-dynamics-models-contain-the-engine.md)
   * [10.47 Is It the Shell or the Parameterisation? The Physics-Injection Grid](docs/paper/10-47-is-it-the-shell-or-the-parameterisation.md)
   * [10.48 What the "Kinematic Violation" Figure Actually Measures](docs/paper/10-48-what-the-kinematic-violation-figure-actually-measures.md)
   * [10.49 Which Documented Constant Does the Telemetry Actually Support?](docs/paper/10-49-which-documented-constant-does-the-telemetry-actually.md)
   * [10.50 Is the Coincidence a Bound or the Support?](docs/paper/10-50-is-the-coincidence-a-bound-or-the.md)
   * [10.51 The Excluded Cell: Bounds Projected onto a State-Output Network](docs/paper/10-51-the-excluded-cell-bounds-projected-onto-a.md)
   * [10.52 Collecting the Missing Branch: the Gravity Gate Under Targeted Excitation](docs/paper/10-52-collecting-the-missing-branch-the-gravity-gate.md)
   * [10.53 The Velocity the Engine Integrates With: a Prediction Target Nobody Trained](docs/paper/10-53-the-velocity-the-engine-integrates-with-a.md)
   * [10.54 Auditing the Physics Prose of Section 4](docs/paper/10-54-auditing-the-physics-prose-of-section-4.md)
   * [10.55 Which Numerical Method Does the Engine Use? The Solver as the Axis](docs/paper/10-55-which-numerical-method-does-the-engine-use.md)
   * [10.56 Canonical Sparse Identification on Every Recording This Repository Has](docs/paper/10-56-canonical-sparse-identification-on-every-recording-this.md)
   * [10.57 What Section 4 Costs: the Corrected Physics, Measured One Line at a Time](docs/paper/10-57-what-section-4-costs-the-corrected-physics.md)
   * [10.58 The Residue of Section 4.1 Is a Clamp, Not a Noise Term](docs/paper/10-58-the-residue-of-section-4-1-is.md)
   * [10.59 The Boundary Channel 10.58 Predicted, Built, and Refuted](docs/paper/10-59-the-boundary-channel-10-58-predicted-built.md)
   * [10.60 The Dashboard at the Top of This File, and the Two Instruments It Needed](docs/paper/10-60-the-dashboard-at-the-top-of-this.md)
   * [10.61 Which Level the Committed Recordings Actually Walk](docs/paper/10-61-which-level-the-committed-recordings-actually-walk.md)
   * [10.62 The Pixel Recording, Re-made From a Level That Is Really Entered](docs/paper/10-62-the-pixel-recording-re-made-from-a-level.md)
   * [10.63 The Held Player Has Two Mechanisms, and the Recordings Mostly Show the Other One](docs/paper/10-63-the-held-player-has-two-mechanisms-and.md)
   * [10.64 The Buffer the Telemetry Reads Is the Title Screen's Own](docs/paper/10-64-the-buffer-the-telemetry-reads-is-the.md)
   * [10.65 Velocity From Frames That Actually Move](docs/paper/10-65-velocity-from-frames-that-actually-move.md)
   * [10.66 The Same Dashboard on Every Stage the Map Will Actually Give, and What Clears Mean](docs/paper/10-66-the-same-dashboard-on-every-stage-the.md)
   * [10.67 Re-running the Map Survey, and the Map a Clear Opens](docs/paper/10-67-re-running-the-map-survey-and-the-map.md)
   * [10.68 The Tile Map Read the Way the Engine Reads It, and a Stage Finished on Screen](docs/paper/10-68-the-tile-map-read-the-way-the.md)
   * [10.69 The Simple Reward, and a Stage No Agent Had Cleared](docs/paper/10-69-the-simple-reward-a-stage-no-agent-had.md)
11. [Complete Reproducibility Guide](docs/paper/11-complete-reproducibility-guide.md)
12. [Scientific Integrity Statement](docs/paper/12-scientific-integrity-statement.md)

---

## Quickstart

```bash
git lfs install          # datasets, checkpoints, figures and savestates are LFS objects
git lfs pull
python -m venv .venv
.venv\Scripts\activate   # Windows: the bundled Libretro core is a Windows DLL
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
make test                # or: smw-pinn test-cov
make reproduce           # the CPU benchmark of Section 8 in minutes
```

The commercial ROM is not distributed: supply your own dump and verify the SHA-1 above
([Section 11.2](docs/paper/11-complete-reproducibility-guide.md#112-environment-setup)).
Everything emulator-free - the identification studies, the operator benchmarks, the conformal
intervals of Section 10.56.1 - runs without it.

## Where things are

- [`docs/paper/`](docs/paper/) - the research document, one page per section.
- [`results/MANIFEST.md`](results/MANIFEST.md) - every published number, indexed artifact-first and
  enforced by CI.
- [`CHANGELOG.md`](CHANGELOG.md) - including the audited self-corrections this repository reports
  instead of quietly applying.
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md),
  [`CITATION.cff`](CITATION.cff).

## License

MIT for the code in `src/`, `tests/` and `scripts/`. The Snes9x Libretro core and the
commercial ROM follow their own terms; the ROM is never distributed.
