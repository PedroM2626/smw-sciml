[← 10.69](10-69-the-simple-reward-a-stage-no-agent-had.md) · [Contents](../../README.md#contents) · [12 →](12-scientific-integrity-statement.md)

## 11. Complete Reproducibility Guide

### 11.1 Consolidated Repository Structure
```
smw-pinn/
├── .github/workflows/ci.yml            # CI: ruff + pytest + coverage on ubuntu-latest
├── configs/                            # YAML benchmark configs (CLI-overridable)
│   ├── base.yaml / benchmark.yaml      # Full 4-model benchmark defaults
│   ├── multiseed.yaml                  # K=10 significance study defaults
│   ├── sample_efficiency.yaml          # Pareto study defaults
│   ├── reproduce.yaml                  # 2-epoch CPU smoke test (`make reproduce`)
│   └── smoke_*.yaml                    # 15 fast emulator-free study configs (`make smoke`)
├── CONTRIBUTING.md                     # Setup, canonical commands, conventions
├── Dockerfile / .dockerignore          # CPU container (CUDA via build-arg)
├── Makefile                            # install / test / lint / reproduce / benchmark
├── data/
│   └── raw/
│       ├── smw_usa.sfc                    # Original retail game ROM (SHA-1 verified)
│       ├── smw_yoshi_island_1.state       # Interactive savestate for Stage A (Yoshi's Island 1)
│       ├── smw_yoshi_island_1_level.state # Verified in-level savestate, entered from the world map (§10.36.3)
│       ├── smw_yoshi_house.state          # Interactive savestate for Stage B (Yoshi's House)
│       ├── smw_gameplay_dataset.npz       # 8,077 genuine interactive transitions (8D)
│       ├── smw_multi_entity_dataset.npz   # 19,702 genuine interactive transitions (12D)
│       ├── smw_pixel_dataset.npz          # 2,720 paired RGB frames + 8D states
│       ├── smw_set_multi_entity_dataset.npz # 3,580 transitions with 12-slot sprite sets
│       ├── smw_tilemap_dataset.npz        # 10,357 genuine transitions with 7x7 tilemaps
│       ├── smw_sprint_dataset.npz         # Excitation-targeted sprint transitions (§10.45)
│       ├── smw_boundary_dataset.npz       # 13,951 transitions with camera, engine mode and WRAM CRC (§10.59)
│       ├── smw_dashboard_run.npz          # One actor episode: probabilities, V(s), operator A, RGB, tiles (§10.60)
│       ├── smw_dashboard_run_smw_map_node_*.npz # The same episode, per stage the map hands over (§10.66)
│       └── smw_map_*.state                # One savestate per walkable node, and per unlocked stage (§10.66)
├── results/
│   ├── MANIFEST.md                        # Artifact index: writer, command, README section
│   ├── *.json                             # one artifact per study; MANIFEST.md above is its index
│   ├── checkpoints/                       # Best trained model & policy weights (.pt)
│   ├── checkpoints_ensemble/              # Deep Ensemble member weights (E=5) (.pt)
│   └── figures/                           # High-resolution benchmark figures (.png) and .gif
├── scripts/
│   ├── inspect_physics.py                 # 60 Hz WRAM telemetry inspector
│   ├── navigate_to_level.py               # Boot & savestate generator (--level 1/2, movement gate)
│   ├── verify_level_entry.py              # A/B the two boot schedules: attract demo vs the map route
│   ├── record_gameplay.py                 # 8D Mario telemetry recorder
│   ├── record_multi_entity_gameplay.py    # 12D Mario + Sprite telemetry recorder
│   ├── record_set_multi_entity_gameplay.py # Full 12-slot sprite-set recorder
│   ├── record_sprint_gameplay.py          # Excitation-targeted sprint recorder (§10.45)
│   ├── record_pixel_gameplay.py           # Paired RGB frame + WRAM recorder
│   ├── record_tilemap_gameplay.py         # 8D + 7x7 tilemap WRAM telemetry recorder
│   ├── scan_scroll_address.py             # Identify the layer-1 scroll from WRAM dumps (10.59)
│   ├── record_boundary_gameplay.py        # 8D + camera + mode + WRAM-CRC recorder (10.59)
│   ├── record_dashboard_run.py            # Actor episode with probabilities, V(s), RGB and operator A (10.60)
│   ├── survey_world_map.py                # Walk the map cursor and press A on every node (10.66, 10.67)
│   ├── unlock_world_map.py                # Fly the clearance controller and read the mode byte (10.66)
│   ├── probe_tile_behaviors.py            # Measure what the engine does with every standard tile id (10.68)
│   ├── probe_sprite_behaviors.py          # Meet every stage sprite and touch it, twice (10.68)
│   └── record_level_dashboards.py         # One episode per stage, flown by the agent that can finish it (10.66, 10.68)
├── src/
│   ├── cli.py                             # Cross-platform entry point (`smw-pinn`, `python -m src.cli`)
│   ├── environment/
│   │   ├── wram.py                        # WRAM address map + the 16-bit tile buffer pages
│   │   ├── tile_behavior.py               # Classify a Map16 id through the measured behaviour table
│   │   ├── tile_behavior_table.py         # The measured table itself (generated; gated against the artifact)
│   │   ├── sprite_behavior.py             # Classify a sprite id through the measured table
│   │   ├── sprite_behavior_table.py       # The measured sprite classes (generated; gated against the artifact)
│   │   ├── bin/snes9x_libretro.dll        # Snes9x Libretro 64-bit core
│   │   ├── snes_emulator.py               # ctypes wrapper: WRAM, sprites, 16-bit tilemap, camera, RGB capture
│   │   ├── scroll_scan.py                 # Rank WRAM words by how well they follow x; scroll axioms
│   │   ├── sprite_sets.py                 # 12-slot sprite → entity-row conversion
│   │   ├── pinn_sim_env.py                # GPU-vectorized World Model simulation environment (8D & 12D)
│   │   └── dataset_loader.py              # PyTorch Dataset and DataLoader loaders
│   ├── perception/
│   │   ├── pixel_encoder.py               # CNN pixel→8D estimator + StateNormalizer
│   │   └── vision_dataset.py              # Paired frame/state dataset + seeded loaders
│   ├── models/
│   │   ├── analytical_kinematics.py       # No-NN closed-form forward model (Baseline A)
│   │   ├── inverse_world_models.py        # MPC wrappers for the identified/symbolic models (§10.44)
│   │   ├── statistical_mlp.py             # Statistical MLP (+ param-matched compact factory)
│   │   ├── statistical_lstm.py            # Statistical LSTM architecture
│   │   ├── pinn_soft.py                   # Soft-Constrained PINN architecture
│   │   ├── pinn_hard_residual.py          # Hard Residual PINN architecture
│   │   ├── pinn_invariant.py              # Translation-Invariant PINN architecture
│   │   ├── pinn_ensemble.py               # Deep Ensemble of Hard PINNs (E=5)
│   │   ├── pinn_multi_entity.py           # Multi-Entity 12D PINN architecture
│   │   ├── pinn_set_multi_entity.py       # Permutation-Invariant Cross-Attention PINN (N Sprites)
│   │   ├── pinn_unified_multimodal.py     # Unified kinematic + tilemap + hazard PINN
│   │   ├── pinn_gravity.py                # Gravity-identified residual PINN (learnable g)
│   │   ├── deeponet.py                    # DeepONet + Physics-Constrained DeepONet operators
│   │   ├── fno.py                         # Fourier Neural Operator (spectral conv baseline)
│   │   ├── residual_dynamics.py           # Increment target + optional hard kinematic shell (10.47)
│   │   ├── output_projection.py           # Bounds projected onto a state-output network (10.51)
│   │   ├── effective_velocity_dynamics.py # The velocity the engine integrates with (10.53)
│   │   ├── neural_ode_dynamics.py       # Learned acceleration field x four solvers (10.55)
│   │   └── tilemap_pinn.py                # Tilemap-conditioned spatial PINN architecture
│   ├── losses/
│   │   └── physics_losses.py              # Analytical physics loss functions
│   ├── utils/
│   │   ├── seed.py                        # Central deterministic seeding
│   │   ├── kinematics.py                  # The two position-integration conventions, named
│   │   ├── typography.py                  # The README's math budget: what stops being `$…$`
│   │   ├── experiment.py                  # TensorBoard + JSONL experiment logger
│   │   ├── logging.py                     # Central stdlib logging helper
│   │   ├── paths.py                       # Repo-root paths + checkpoint/figure helpers (CWD-safe)
│   │   ├── provenance.py                  # `_meta` provenance stamps + strict JSON writer
│   │   └── config.py                      # YAML config + CLI-override loader
│   ├── training/
│   │   ├── trainer.py                     # Training loop with Early Stopping & LR scheduler
│   │   ├── train_multi_entity.py          # Supervised training for hazard_net on 12D WRAM data
│   │   ├── train_tilemap.py               # Supervised training for TilemapPINNDynamics
│   │   ├── benchmark_experiment.py        # Main comparative benchmark (--config, --matched-baseline)
│   │   ├── dyna_ppo.py                    # Amortized Policy Optimization (Dyna-PPO)
│   │   ├── dyna_ppo_sprites.py            # Multi-Entity 12D Policy Optimization
│   │   ├── distill_mpc_policy.py          # MPC trajectory distillation via imitation learning
│   │   ├── model_free_ppo.py              # Canonical Model-Free PPO baseline on real SNES
│   │   ├── train_dagger.py                # Interactive DAgger imitation training
│   │   ├── train_unified_ppo.py           # Unified Dyna-PPO in PINN GPU simulator
│   │   ├── train_pixel_estimator.py       # CNN pixel→state supervised training
│   │   ├── train_terminal_value.py        # TD-MPC terminal value (MC regression on hw log)
│   │   ├── train_unified_multimodal.py    # Joint tilemap+hazard training (two datasets)
│   │   ├── train_set_multi_entity.py      # Supervised training on 12-slot sprite sets
│   │   └── online_mbpo.py                 # Closed-loop Online MBPO & Safe MBPO engine
│   ├── planning/
│   │   ├── mpc_planner.py                 # GPU-vectorized CEM / Random Shooting MPC planner
│   │   ├── terrain_runner.py              # The rule-based agent over the measured tables (10.68)
│   │   ├── global_planner.py              # A* occupancy grid + waypoints + hierarchical MPC
│   │   ├── terminal_value.py              # TD-MPC terminal value net + objective
│   │   ├── tilemap_mpc.py                 # TilemapPINN→MPC adapter (static-map approx)
│   │   └── differentiable_pinn_planner.py # First-order gradient control through Hard PINN
│   ├── inverse/
│   │   ├── parameter_identification.py    # 7-constant inverse problem + Laplace/MCMC posterior (§10.40)
│   │   ├── structure_selection.py         # Nested-template selection of clamp/gate structures (§10.43.9)
│   │   ├── sindy_identification.py        # Polynomial dictionary, ridge and sequential thresholding (§10.56)
│   │   └── symbolic_regression.py         # GP law discovery, excitation normalisation, constant probes (§10.43)
│   └── evaluation/
│       ├── analytical_baselines.py        # No-NN baseline + oracle-MPC upper bound (§10.37)
│       ├── ablation_benchmark.py          # PINN constraint ablation (soft vs hard vs MLP)
│       ├── evaluate_dagger_snes.py        # DAgger policy hardware evaluation
│       ├── rollout_evaluator.py           # Rollout evaluator (+ multi-start statistics)
│       ├── per_variable_metrics.py        # Per-channel MSE/MAE/R² + contact accuracy/F1
│       ├── sample_efficiency_benchmark.py # Sample efficiency Pareto benchmark script
│       ├── deeponet_benchmark.py          # DeepONet neural-operator study (10.41, CI-safe)
│       ├── operator_benchmark.py          # Operator family: PC-DeepONet + FNO (10.42, CI-safe)
│       ├── inverse_transfer_benchmark.py  # Parameter identification + zero-shot transfer (10.40, CI-safe)
│       ├── symbolic_inverse_benchmark.py  # Symbolic-regression inverse study (10.43, CI-safe)
│       ├── symbolic_tilemap_residual_benchmark.py # Terrain-conditioned residual control (10.43.8)
│       ├── symbolic_engine_ablation_benchmark.py # Three-engine discovery ablation (10.43.9)
│       ├── inverse_model_mpc_benchmark.py # Closed-loop inverse-model MPC comparison (10.44, hardware)
│       ├── sprint_excitation_benchmark.py # Ceiling measured on two recordings (10.45)
│       ├── learned_structure_probe_benchmark.py # Fixed-point/gate probes on learned models (10.46)
│       ├── operator_physics_injection_benchmark.py # Target x mechanism x family grid (10.47)
│       ├── physics_injection_mpc_benchmark.py # One study's arms flown on the console, by --study (10.47, 10.51, 10.53, 10.55, 10.57.1; hardware)
│       ├── rollout_diagnostics.py           # Violation rate split into consistency/smoothness/bounds (10.48)
│       ├── kinematic_metric_decomposition_benchmark.py # Re-scores every model with the split metric (10.48)
│       ├── velocity_class_benchmark.py      # Which documented speed constant the telemetry reaches (10.49)
│       ├── plateau_provenance_benchmark.py  # Plateau vs training-support truncation (10.50)
│       ├── projection_cell_benchmark.py     # The state x hard output-projection cell (10.51)
│       ├── gate_excitation_benchmark.py     # The gravity gate measured on three recordings (10.52)
│       ├── effective_velocity_benchmark.py  # Convention x family x mechanism grid (10.53)
│       ├── physics_claim_audit.py           # Audits README section 4 against code (10.54)
│       ├── neural_ode_integrator_benchmark.py # Learned field x four solvers (10.55)
│       ├── sindy_identification_benchmark.py  # Canonical SINDy on five recordings (10.56)
│       ├── corrected_physics_ablation.py    # Prices the section-4 corrections (10.57)
│       ├── residue_process_study.py           # Is the section-4.1 residue noise, a jump or a clamp? (10.58, emulator-free)
│       ├── boundary_channel_study.py          # The camera channel 10.58 predicted, tested and refuted (10.59, emulator-free)
│       ├── pixel_frame_probe.py                   # Do the committed RGB frames scroll with the player (10.31.1)
│       ├── dashboard_video.py                # Render the README hero: RGB + tilemap + probabilities (10.60)
│       ├── dashboard_study.py                # Which actor flies the hero, and the frame-buffer defect (10.60)
│       ├── stage_identity_study.py             # Which terrain the committed recordings actually walk (10.61)
│       ├── pixel_level_recheck.py              # The pixel recording re-made from the level entry (10.62)
│       ├── pause_hygiene_study.py               # The held player split: death apex or message box (10.63)
│       ├── forced_mode_buffer_study.py          # Whose tile buffer the telemetry reads (10.64)
│       ├── pixel_retrain_study.py                # The 2x2 that re-measures 10.31 on moving frames (10.65)
│       ├── level_dashboard_study.py              # The dashboard on every stage the map gives (10.66)
│       ├── world_map_recheck_study.py            # The survey run again, and the map a clear opens (10.67)
│       ├── perception_correction_study.py         # The measured tile/sprite tables and the stage finished on screen (10.68)
│       ├── multiseed_benchmark.py         # K=10 multi-seed significance benchmark (+Cohen's dz)
│       ├── mbrl_mpc_benchmark.py          # Closed-loop MBRL benchmark on SNES emulator
│       ├── evaluate_multi_entity_mpc.py   # Autonomous 12D MPC closed-loop evaluation on SNES
│       ├── evaluate_distilled_policy_snes.py # Distilled amortized policy hardware benchmark
│       ├── evaluate_extended_navigation.py # Extended 1,000+ px hardware navigation benchmark
│       ├── evaluate_policy_snes.py        # Zero-shot Model-to-Real transfer benchmark on SNES
│       ├── evaluate_sprites_snes.py       # Dynamic sprite perception and Rex evasion benchmark
│       ├── evaluate_multi_entity_snes.py  # End-to-end 12D zero-shot hardware benchmark
│       ├── benchmark_computational_efficiency.py # Comprehensive hardware efficiency profiling
│       ├── cross_level_benchmark.py       # Out-of-distribution cross-stage generalization
│       ├── evaluate_cross_level_control.py # Zero-shot closed-loop control on Stage B
│       ├── evaluate_full_level_clearance.py # Full stage clearance benchmark
│       ├── diagnose_obstacle_1000.py        # Formal X~1000 bottleneck diagnosis
│       ├── mpc_reflex_ablation.py           # Pure vs reflexive MPC honesty ablation
│       ├── evaluate_tilemap_mpc.py          # Terrain-anticipating closed-loop MPC
│       ├── evaluate_pixel_mpc.py            # Pixel→estimate→MPC closed loop
│       ├── evaluate_hierarchical_mpc.py     # A* global + local MPC (--value-ckpt TD-MPC)
│       ├── plot_learning_curves.py          # Model-Free vs Dyna comparison figure
│       ├── spatial_holdout_benchmark.py     # OOD-with-danger holdout (X>700, CI-safe)
│       ├── render_level_clearance_video.py # Telemetry HUD video/GIF renderer
│       └── render_comparison_animation.py # Synchronized trajectory animation generator
├── tests/ (340+ tests: unit + regression + smoke + emulator-guarded integration)
│   ├── conftest.py                        # requires_emulator guard (Windows DLL)
│   ├── test_losses.py                     # Unit tests for physics loss functions
│   ├── test_models.py                     # Unit tests for tensor shapes and forward passes
│   ├── test_dataset_loader.py             # Split disjointness, leakage guard, seeded shuffle
│   ├── test_trainer.py                    # Convergence, checkpointing, early stopping
│   ├── test_rollout_multistart.py         # Multi-start stats + per-variable metrics
│   ├── test_matched_baseline.py           # ~10k param-parity pair (MLP vs Hard PINN)
│   ├── test_metrics_regression.py         # Guards published numbers (fails on silent decay)
│   ├── test_config.py                     # YAML load + CLI-override semantics
│   ├── test_emulator_guard.py             # Platform guard unit tests
│   ├── test_mpc_planner.py                # Unit tests for MPC trajectory planner
│   ├── test_dyna_ppo.py                   # Unit tests for PINNVectorEnv & Dyna-PPO agent
│   ├── test_sprites.py                    # Unit tests for WRAM sprite extraction & hazard distance
│   ├── test_multi_entity.py               # Unit tests for Multi-Entity 12D kinematics & env
│   ├── test_set_multi_entity.py           # Unit tests for Permutation-Invariant Multi-Entity PINN
│   ├── test_computational_efficiency.py   # Unit tests for profiling calculations
│   ├── test_pinn_invariant.py             # Unit tests for spatial translation equivariance
│   ├── test_pinn_ensemble.py              # Unit tests for ensemble predictions & epistemic variance
│   ├── test_model_free_ppo.py             # Unit tests for real-emulator environment wrapper
│   ├── test_online_mbpo.py                # Unit tests for real replay buffer & sampling
│   ├── test_tilemap.py                    # Unit tests for WRAM tilemap extraction & TilemapPINN
│   ├── test_differentiable_planner.py     # Unit tests for gradient-based PINN planner
│   ├── test_unified_multimodal.py         # Unit tests for unified multimodal PINN
│   ├── test_cross_level_control.py        # Unit tests for Stage-B control utilities
│   ├── test_seed.py                       # Unit tests for deterministic seeding
│   ├── test_kinematics_modes.py           # The corrected modes exist; the defaults did not move
│   ├── test_typography.py                 # What converts out of math, and what must not
│   ├── test_experiment.py                 # Unit tests for experiment logger
│   ├── test_pixel_perception.py           # Frame conversion, CNN estimator, vision data
│   ├── test_global_planner.py             # A*, waypoints, hierarchical control
│   ├── test_tdmpc_reflex.py               # Terminal value, reflex rules, diagnose math
│   ├── test_orphans_gravity.py            # Tilemap wrapper, sprite rows, gravity ID
│   ├── test_paths.py                      # Repo-root paths work from any CWD
│   ├── test_dependency_parity.py          # requirements.txt / pyproject / lockstep pins
│   ├── test_analytical_baselines.py       # Closed-form kinematics + oracle MPC
│   ├── test_results_manifest.py           # Artifact catalog, freshness, README headlines
│   ├── test_deeponet.py                   # Unit tests for the DeepONet operator baselines
│   ├── test_fno.py                        # Unit tests for the Fourier Neural Operator
│   ├── test_symbolic_regression.py        # GP law banks, probes, analytic-law parity (§10.43)
│   ├── test_symbolic_tilemap_residual.py  # Terrain descriptors, placebo, verdict rule (§10.43.8)
│   ├── test_structure_selection.py        # Template fits, BIC/tail selection, agreement flag (§10.43.9)
│   ├── test_symbolic_engine_ablation.py   # Bound probe, engine verdicts, telemetry reading (§10.43.9)
│   ├── test_sprint_excitation.py          # Excitation profile and ceiling verdict (§10.45)
│   ├── test_learned_structure_probe.py    # Ceiling classification and traction plausibility (§10.46)
│   ├── test_residual_dynamics.py          # Shared shell, parameterisation parity and the 10.47 grid plan
│   ├── test_effective_velocity_dynamics.py # Convention modes, invariance and the ladder
│   ├── test_neural_ode_dynamics.py        # Solver algebra, coefficients and the bounded step
│   ├── test_sindy_identification.py       # Dictionary, thresholding in physical units, recovery
│   ├── test_inverse_world_models.py       # Inverse-model MPC wrappers + figure/agreement (§10.44)
│   ├── test_hardware_loops.py             # Emulator-guarded end-to-end hardware entry points
│   ├── test_smoke_runs.py                 # Every configs/smoke_*.yaml is accepted + runs
│   └── test_english_only.py               # Repo text stays English-only
├── pyproject.toml                         # Python package and pytest configuration
├── README.md                              # Complete experimental documentation and benchmark report
├── CONTRIBUTING.md                        # Setup, canonical commands, conventions
├── CITATION.cff                           # Citation metadata
├── LICENSE                                # MIT license
└── requirements.txt                       # Project dependency manifest
```

### 11.2 Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate
# CUDA 12.1 build (RTX 4070 reference hardware):
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# CPU-only fallback:
# pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Editable install (enables `from src...` imports without sys.path hacks):
pip install -e ".[dev]"
# Large binaries/datasets are versioned via Git-LFS:
git lfs install
git lfs pull
```

> **Determinism:** all entry points seed Python/NumPy/PyTorch via `src/utils/seed.py:set_global_seed(seed, deterministic=True)` (cuDNN deterministic, `PYTHONHASHSEED` fixed). Pass `--non-deterministic` only when benchmarking raw throughput.

### 11.3 Running Automated Unit Tests
```bash
python -m pytest tests/ -v
```

### 11.4 Experiment Tracking (TensorBoard / JSONL)
Training scripts log per-epoch metrics to `runs/<experiment>_<timestamp>/` (`metrics.jsonl` + `hparams.json` + TensorBoard events) via `src/utils/experiment.py:ExperimentLogger`:
```bash
python -m src.training.benchmark_experiment --experiment-name benchmark_mlp_vs_pinn
tensorboard --logdir runs
# Disable TensorBoard (keep JSONL): --no-tensorboard
# Mirror to wandb (requires pip install -e ".[wandb]"): --wandb
```

### 11.5 Reproducing Benchmarks & MBRL Evaluations

> **Run entry points as modules** (`python -m src.evaluation.foo`), never as file
> paths (`python src/evaluation/foo.py`): the package uses absolute `from src...`
> imports, so the file-path form fails with `ModuleNotFoundError: No module named 'src'`.
> On Windows, where `make` is usually absent, `pip install -e ".[dev]"` also provides the
> equivalent console script: `smw-pinn benchmark`, `smw-pinn multiseed`,
> `smw-pinn baselines`, or generically `smw-pinn run src.evaluation.<module> [flags]`.
> Emulator-in-the-loop commands additionally need the Libretro core and your own ROM dump;
> their locations are resolved by `src/utils/paths.py` and can be overridden with the
> `SMW_ROM` / `SMW_CORE` / `SMW_DATA_DIR` environment variables (see section 11.2).

```bash
# 1. Main comparative benchmark across all 4 architectures (single-seed):
python -m src.training.benchmark_experiment

# 2. Sample efficiency Pareto curve benchmark (N = 200 to 5,000):
python -m src.evaluation.sample_efficiency_benchmark

# 3. Multi-seed statistical significance benchmark (K = 10 seeds, t-test + Wilcoxon + Cohen's dz):
python -m src.evaluation.multiseed_benchmark

# 4. Closed-loop Model-Based RL (MPC) benchmark in real SNES console emulator:
python -m src.evaluation.mbrl_mpc_benchmark

# 5. Amortized Policy Optimization (Dyna-PPO) and Zero-Shot Model-to-Real Transfer:
python -m src.training.dyna_ppo
python -m src.evaluation.evaluate_policy_snes

# 6. Dynamic WRAM sprite perception and Rex evasion benchmark:
python -m src.evaluation.evaluate_sprites_snes

# 7. Canonical Model-Free PPO baseline on real SNES console emulator:
python -m src.training.model_free_ppo

# 8. Deep PINN Ensemble (E=5) training & epistemic uncertainty quantification:
python -m src.models.pinn_ensemble

# 9. Full closed-loop Online MBPO (Model-Based Policy Optimization):
python -m src.training.online_mbpo

# 10. Multi-Entity 12D PINN training & autonomous hazard evasion:
python -m src.training.dyna_ppo_sprites
python -m src.evaluation.evaluate_multi_entity_snes

# 11. Safe Closed-Loop MBPO with Deep Ensemble & Epistemic Truncation:
python -m src.training.online_mbpo --safe

# 12. Hardware & computational efficiency profiling suite:
python -m src.evaluation.benchmark_computational_efficiency

# 13. Cross-stage zero-shot generalization benchmark (Yoshi's House):
python -m src.evaluation.cross_level_benchmark

# 14. Synchronized multi-model visualization and animation generation:
python -m src.evaluation.render_comparison_animation

# 15. Record genuine 12D Multi-Entity dataset from SNES WRAM:
python -m scripts.record_multi_entity_gameplay

# 16. Supervised training of hazard dynamics in MultiEntityPINNDynamics:
python -m src.training.train_multi_entity

# 17. Autonomous Multi-Entity MPC Planning on live SNES hardware (782 px Rex evasion):
python -m src.evaluation.evaluate_multi_entity_mpc

# 18. Record genuine 8D + 7x7 tilemap dataset from SNES WRAM ($7E:C800):
python -m scripts.record_tilemap_gameplay

# 19. Supervised training of TilemapPINNDynamics on genuine stage geometry:
python -m src.training.train_tilemap

# 20. Distill MPC expert trajectories into an ultra-fast amortized policy (2,900 FPS):
python -m src.training.distill_mpc_policy
python -m src.evaluation.evaluate_distilled_policy_snes

# 21. Autonomous Extended Hardware Navigation on real SNES (1,016+ px progress):
python -m src.evaluation.evaluate_extended_navigation

# 22. Zero-Shot Closed-Loop Control Benchmark on Unseen Stage B (Yoshi's House, 5-seed mean +/- std):
python -m src.evaluation.evaluate_cross_level_control

# 23. Train Unified Dyna-PPO inside PINN GPU Simulator (>14,000 FPS):
python -m src.training.train_unified_ppo

# 24. Full Stage Clearance Benchmark on Live SNES Hardware (PPO vs. DAgger vs. MPC):
python -m src.evaluation.evaluate_full_level_clearance --controller dagger
python -m src.evaluation.evaluate_full_level_clearance --controller ppo

# 25. Render High-Resolution Video (MP4) and Animated GIF with Telemetry HUD:
python -m src.evaluation.render_level_clearance_video

# 26. Diagnose the X~1000 bottleneck (tile gaps + sprite census, real hardware):
python -m src.evaluation.diagnose_obstacle_1000

# 27. Honesty ablation: pure vs reflexive MPC (600 frames each, same savestate):
python -m src.evaluation.mpc_reflex_ablation

# 28. Terrain-anticipating Tilemap-MPC closed loop (no reflexes):
python -m src.evaluation.evaluate_tilemap_mpc

# 29. Joint training of Unified Multimodal PINN (tilemap + hazard datasets):
python -m src.training.train_unified_multimodal

# 30. Full 12-slot sprite-set recording + Set-Multi-Entity training:
python -m scripts.record_set_multi_entity_gameplay
python -m src.training.train_set_multi_entity

# 31. Pixel perception: record frames, train estimator, close pixel→MPC loop:
python -m scripts.record_pixel_gameplay
python -m src.training.train_pixel_estimator
python -m src.evaluation.evaluate_pixel_mpc

# 32. Hierarchical A* + MPC (optional TD-MPC terminal value):
python -m src.evaluation.evaluate_hierarchical_mpc

# 33. TD-MPC terminal value fitting + spatial-holdout OOD (CI-safe, no emulator):
python -m src.training.train_terminal_value
python -m src.evaluation.spatial_holdout_benchmark

# 34. Model-Free vs Dyna learning-curve figure (from committed artifacts):
python -m src.evaluation.plot_learning_curves

# 35. Reference baselines: zero-parameter engine rules (no emulator) + oracle-model
#     MPC upper bound (needs the emulator; use --no-hardware to skip it):
python -m src.evaluation.analytical_baselines

# 36. Report where the ROM, core and datasets actually resolve to (diagnostics):
python -m src.cli install-info

# 37. Closed-loop reproduction audit: re-run the 10.6 protocol on several seeds
#     without touching mbrl_mpc_metrics.json (see 10.38.2):
python -m src.evaluation.mbrl_mpc_benchmark --reproduction-check --repeats 3

# 38. Preamble probe: quantify how much progress the episode start alone is worth
#     (see 10.38.1 - the committed savestate restores into mode 0x08):
python -m src.evaluation.mbrl_mpc_benchmark --preamble-probe

# 39. Pixel front-end: record paired (frame, WRAM) data, train the estimator and
#     close the loop blind to WRAM (10.31), then the hierarchical A*+MPC run (10.32):
python -m scripts.record_pixel_gameplay
python -m src.training.train_pixel_estimator
python -m src.evaluation.evaluate_pixel_mpc
python -m src.evaluation.evaluate_hierarchical_mpc

# 40. Yoshi's Island 2 capture attempt with the documented recipe and evidence
#     gate - it raises and writes results/yi2_capture_attempt.json (10.36):
python -m scripts.navigate_to_level --level 2

# 41. Physics-Informed Model-Free RL (PIML-MFRL): model-free PPO on the real SNES
#     console with the physics-informed critic (A), CBF actor filter (B) and
#     action-violation penalty (C) enabled (10.39). Needs core + ROM.
python -m src.training.piml_mfrl --config configs/piml_mfrl.yaml

# 42. PIML-MFRL per-mechanism ablation study (model-free baseline + A / B / C / A+B+C,
#     3 seeds), which writes results/piml_mfrl_metrics.json + the comparison figure (10.39.1). Needs core + ROM.
python -m src.evaluation.piml_mfrl_study --seeds 42,43,44 --total-timesteps 10000

# 43. Physics parameter identification (the inverse problem) + zero-shot control transfer
#     (10.40). Emulator-free: runs on CPU from the recorded dataset, so it is a CI-safe study.
python -m src.evaluation.inverse_transfer_benchmark

# 44. DeepONet neural-operator baseline under the unified protocol (10.41).
#     Emulator-free: writes results/deeponet_benchmark_metrics.json.
python -m src.evaluation.deeponet_benchmark

# 45. Neural-operator family study: DeepONet + Physics-Constrained DeepONet + FNO
#     (10.42). Emulator-free: writes results/operator_benchmark_metrics.json.
python -m src.evaluation.operator_benchmark

# 46. Symbolic regression as an inverse-problem method (10.43): genetic programming
#     discovers the one-step laws, a probe stage reads the constants back out, and the
#     result is compared against the 10.40 parametric identification, the 8.2 guarantee
#     metrics and the published learned models. Emulator-free CPU study (~15 min):
#     writes results/symbolic_inverse_metrics.json.
python -m src.evaluation.symbolic_inverse_benchmark

# 47. Terrain-conditioned residual discovery (10.43.8): the 10.43.5 grey-box control with the
#     recorded 7x7 block buffer available to the search, against a shuffled-geometry placebo.
#     Emulator-free: writes results/symbolic_tilemap_residual_metrics.json.
python -m src.evaluation.symbolic_tilemap_residual_benchmark

# 48. Three-engine discovery ablation (10.43.9): the same inverse problem through bounded-terminal
#     gplearn, PySR (unbounded, numerically optimised constants; optional `pip install -e
#     ".[symbolic]"`, recorded as unavailable otherwise) and nested-template BIC/tail selection,
#     plus a plateau-mechanism specificity control and budget sweeps for both tree engines.
#     Emulator-free CPU study (~2 h, dominated by the two budget sweeps): writes
#     results/symbolic_engine_ablation_metrics.json.
python -m src.evaluation.symbolic_engine_ablation_benchmark

# 49. Closed-loop control with inverse-problem world models (10.44): one CEM-MPC planner,
#     objective, savestate and frame budget, driven by the established WRAM rules, the 10.40
#     identified constants, the 10.43 discovered laws and the published Hard Residual PINN.
#     Requires the Libretro core and ROM dump (11.2); writes results/inverse_model_mpc_metrics.json.
python -m src.evaluation.inverse_model_mpc_benchmark

# 50. Excitation-targeted ceiling measurement (10.45). Step 1 records a dataset whose only
#     job is to saturate the speed bound, driven by the 10.37 established-rules MPC with a
#     speed-weighted objective (needs core + ROM, ~15 min). Step 2 runs the three engines of
#     10.43.9 over both recordings and reports whether the ceiling becomes identifiable.
python scripts/record_sprint_gameplay.py --episodes 12 --frames-per-episode 800
python -m src.evaluation.sprint_excitation_benchmark

# 51. Structural probes on the learned dynamics models (10.46): the 10.43 fixed-point and
#     gravity-gate probes applied to the committed MLP / Soft / Hard PINN / DeepONet /
#     PC-DeepONet / FNO checkpoints, plus identification re-run through each of them as a
#     surrogate forward. Emulator-free (~1 min): writes results/learned_structure_probe_metrics.json.
python -m src.evaluation.learned_structure_probe_benchmark

# 52. Physics-injection grid (10.47): family x target x mechanism, 17 arms over five seeds,
#     all emulator-free (~27 min). `--arms MLP` runs one family, `--epochs` and `--seeds`
#     cut the budget, `--no-checkpoints` skips publishing the primary-seed weights.
#     Writes results/operator_physics_injection_metrics.json (+ `make physics-injection`).
python -m src.evaluation.operator_physics_injection_benchmark

# 53. The same grid arms flown on the real console with the published 10.44 planner (~18 min,
#     hardware). Writes results/physics_injection_mpc_metrics.json; exits with a diagnostic
#     and writes nothing without the Libretro core and ROM.
python -m src.evaluation.physics_injection_mpc_benchmark --seeds 42,43,44,45,46

# 54. The 10.46 structural probes applied to the 10.47 arms (~1 min, emulator-free). Writes a
#     separate artifact, so `--registry published` (the 10.46 default) is never overwritten.
python -m src.evaluation.learned_structure_probe_benchmark --registry grid

# 55. Decompose the published rollout violation figure into integration residual,
#     velocity-jump rate and bound exceedance, for every committed model and for the
#     recorded telemetry (10.48). Emulator-free (~1 min); `--num-starts`/`--horizon`
#     control the budget; writes results/kinematic_metric_decomposition_metrics.json.
python -m src.evaluation.kinematic_metric_decomposition_benchmark

# 56. Which of Section 4's documented speed constants the recordings actually reach
#     (10.49): the velocity envelope of the four recordings that study covers, against the walk/run/P-meter
#     classes, the vertical window, and the integration-convention test. Emulator-free,
#     ~2 s; writes results/velocity_class_metrics.json.
python -m src.evaluation.velocity_class_benchmark

# 57. Follow a learned plateau with the data it was shown (10.50): train each family on
#     training splits truncated at four velocity caps and re-probe over the full range.
#     Emulator-free (~15 min); `--caps`, `--seeds` and `--epochs` cut the budget; writes
#     results/plateau_provenance_metrics.json.
python -m src.evaluation.plateau_provenance_benchmark

# 58. The cell 10.47 excluded (10.51): fit the state-output networks with the bounds
#     projected onto their output (~6 min), then fly them against the 10.42 shell on the
#     console (~12 min, hardware). Writes results/projection_cell_metrics.json and
#     results/projection_cell_mpc_metrics.json.
python -m src.evaluation.projection_cell_benchmark
python -m src.evaluation.physics_injection_mpc_benchmark --study projection

# 59. Collect and measure the missing branch of the gravity gate (10.52): step 1 records
#     jump-excited WRAM telemetry (hardware, ~15 min), step 2 measures the tiers on the
#     published, sprint and jump recordings with the engines of 10.43.9 (emulator-free,
#     ~20 min; `--only published_gameplay` and the GP budget flags cut it).
python scripts/record_jump_gameplay.py --episodes 10 --frames-per-episode 700
python -m src.evaluation.gate_excitation_benchmark

# 60. Train the prediction target the repository never had (10.53): the velocity the
#     engine integrates with. Step 1 is the 27-cell convention x family x mechanism grid
#     (emulator-free, ~14 min); step 2 flies the nine unconstrained arms beside their own
#     published-convention twins (hardware, ~12 min).
python -m src.evaluation.effective_velocity_benchmark --seeds 42,43,44,45,46
python -m src.evaluation.physics_injection_mpc_benchmark --study effective

# 61. Audit README section 4 against the code that implements it and the telemetry it
#     describes (10.54): emulator-free and seconds long.
python -m src.evaluation.physics_claim_audit

# 62. Ask which numerical method the engine is (10.55): read the same network as a
#     continuous acceleration field and integrate it with four named solvers. Step 1
#     is the 24-cell family x solver x bounding grid (emulator-free, ~45 min); step 2
#     flies the twelve unconstrained arms on the console (hardware, ~30 min).
python -m src.evaluation.neural_ode_integrator_benchmark --seeds 42,43,44,45,46
python -m src.evaluation.physics_injection_mpc_benchmark --study ode

# 63. Identify the same laws with the canonical SciML estimator instead of searching
#     for them (10.56): dictionary, ridge, sequential thresholding, five recordings,
#     two losses. Emulator-free and ~20 s.
python -m src.evaluation.sindy_identification_benchmark --alpha 0.05

# 64. Price the three corrections README 10.54 records as prose-and-code disagreement (10.57):
#     the 4.1 integration convention, the 4.3.5 ground rule and the predicate's velocity bound,
#     each changed one at a time over five seeds. Emulator-free, and its own artifact accounts
#     for 715 s of training across the 80 fits; pass --save-checkpoints to publish the eight arms
#     the console leg flies. Writes results/corrected_physics_ablation_metrics.json.
#     `make corrected-physics` does both, the second step on real hardware: 11 controllers x 5
#     CEM seeds, 55 episodes, 16 min on this host with the GPU shared with another training job
#     (needs core + ROM).
python -m src.evaluation.corrected_physics_ablation

# 65. Decide what the residue of section 4.1 is: a lattice with a memory, a jump measure or
#     a state constraint (10.58). Six recordings, 55,085 transitions, four candidate
#     propagators scored on the recorded trajectories. Emulator-free, ~3 min.
python -m src.evaluation.residue_process_study

# 66. Identify the horizontal camera from the console instead of from a memory map (10.59): dump
#     all 128 KB of WRAM every frame of a scripted run, rank the 16-bit words by how well they
#     follow Mario's x, keep the ones satisfying the scroll axioms. Needs core + ROM; ~1 min.
python scripts/scan_scroll_address.py

# 67. Record the camera, the engine mode and a WRAM CRC beside the published state, then test
#     whether a boundary channel closes the residue (10.59). The recorder needs core + ROM (~7 s);
#     the study is emulator-free and also re-runs the repeat test on the six older recordings.
python scripts/record_boundary_gameplay.py
python -m src.evaluation.boundary_channel_study

# 68. Audit the pixel recording of 10.31 without a model: align every consecutive frame pair by the
#     integer horizontal shift that best matches it and compare that with the change in the recorded
#     x, with a synthetic control that proves the alignment sees a scroll when one is made.
#     Emulator-free; reads data/raw/smw_pixel_dataset.npz. Writes
#     results/pixel_frame_probe_metrics.json and the paragraph README 10.31.1 quotes.
python -m src.evaluation.pixel_frame_probe

# 69. A/B the level-entry schedules on the console (§10.36.3): the old START/B holds reach a rendered
#     level whose player moves 0.0 px, the world-map route reaches one that travels 866.8 px with the
#     camera following. Needs core + ROM; ~1 min. Writes results/level_entry_verification.json, and
#     `python scripts/navigate_to_level.py --level 1` now saves only a state that passes the
#     control-response gate.
python scripts/verify_level_entry.py

# 70. Record the dashboard episode (10.60): probe both actor checkpoints on the verified
#     level-entry savestate, keep the arm the rule selects, and record one episode with the actor's
#     per-button probabilities, the terminal-value critic's V(s), the buttons the console actually
#     received, the tile patch and camera, an RGB frame, the section-4.1 residue of every step and
#     the operator's A presses. Needs core + ROM; ~20 s. Writes
#     data/raw/smw_dashboard_run.npz, results/dashboard_run_metrics.json and
#     results/dashboard_controller_selection.json.
python scripts/record_dashboard_run.py

# 71. Render the hero from that recording: the emulator's own picture beside the tilemap, with the
#     probability, value, residue and progress panels. Emulator-free (it reads the recording);
#     ~50 s for 634 frames, and the GIF is derived from the MP4 by the bundled ffmpeg. Writes
#     results/figures/dashboard_run.mp4, results/figures/dashboard_run.gif and
#     results/dashboard_video_metrics.json.
python -m src.evaluation.dashboard_video

# 72. Gather the three artifacts into the section this file prints (10.60), cut the message-box
#     evidence figure out of the committed recording, and re-measure the forced-mode picture claim
#     of 10.31/10.36.3 with the corrected decoder (the last step needs core + ROM, ~10 s).
python -m src.evaluation.dashboard_study

# 73. Settle which level the committed telemetry recordings walk (10.61): reconstruct "what is under
#     Mario" per absolute tile column from the recorded 7x7 patches, then read the same profile live
#     from the forced-mode session it came from and from the verified world-map level entry, under an
#     identical scripted walk. Needs core + ROM; ~20 s. Writes results/stage_identity_metrics.json.
python -m src.evaluation.stage_identity_study

# 74. Re-make the pixel recording from the verified level entry (10.62) and score it against the
#     camera rather than the player. The recorder needs core + ROM (~15 s); the recheck is
#     emulator-free. Writes data/raw/smw_pixel_level_dataset.npz (a NEW file - the published
#     smw_pixel_dataset.npz is left as §10.31 measured it) and
#     results/pixel_level_recheck_metrics.json.
python scripts/record_pixel_gameplay.py --state data/raw/smw_yoshi_island_1_level.state     --output data/raw/smw_pixel_level_dataset.npz --clear-boxes --episodes 12 --frames 700
python -m src.evaluation.pixel_level_recheck

# 75. Split 10.59's held-player signature by the velocity byte it freezes with (10.63), on three
#     arms of one recorder: the published boundary file, the entered level with no hygiene, and the
#     same walk with the hold cleared by an operator A pulse. The two new arms need core + ROM (~35 s
#     each); the study is emulator-free and compares them channel by channel. Writes
#     data/raw/smw_boundary_level_dataset.npz, data/raw/smw_boundary_level_clear_dataset.npz and
#     results/pause_hygiene_metrics.json.
python scripts/record_boundary_gameplay.py --state data/raw/smw_yoshi_island_1_level.state     --output data/raw/smw_boundary_level_dataset.npz
python scripts/record_boundary_gameplay.py --state data/raw/smw_yoshi_island_1_level.state     --clear-boxes --output data/raw/smw_boundary_level_clear_dataset.npz
python -m src.evaluation.pause_hygiene_study

# 76. Identify whose tile buffer the telemetry reads (10.64): sample all 27 rows of every absolute
#     level column in three sessions - the title screen with its attract demo, the published
#     forced-mode preamble, and the entered level of 10.36.3 - and score them by whole-column identity
#     of raw tile IDs, by 10.61's class profile, and by the translevel byte. Needs core + ROM, ~1 min.
#     Writes results/forced_mode_buffer_metrics.json.
python -m src.evaluation.forced_mode_buffer_study

# 77. Re-measure 10.31's velocity conclusion as a 2x2 of recording (title composite, scrolling level)
#     by input (one frame, two frames), with the published run reused as the parity cell. Emulator-free;
#     ~4 min for the three new cells on this host. Writes pixel_estimator_retrain_* checkpoints and
#     metrics beside the published ones (never over them) and results/pixel_retrain_metrics.json.
python -m src.evaluation.pixel_retrain_study

# 78. Survey the world map (10.66): walk the cursor in all four directions from the node 10.36.3
#     measured, replay a recorded direction sequence to reach each position again, press A on every
#     one, and save the state of each session the engine hands over. Needs core + ROM, ~3 min.
#     Writes results/world_map_survey.json and data/raw/smw_map_node_*.state.
python -m scripts.survey_world_map

# 79. Ask whether anything clears (10.66): fly the published clearance controller at the entered
#     stage, sample $7E:0100 every frame, and call a run a clear only when the engine reaches the map
#     without passing through the death mode 0x0B and the cursor moves off the node it entered.
#     Seeded, the clearing flight is re-flown once and must reproduce to the subpixel. Needs core +
#     ROM, ~9 min with --max-frames 8000. Writes results/world_map_clearance.json and, for each
#     stage it enters, a savestate under data/raw.
python -m scripts.unlock_world_map --rounds 2 --attempts 2 --max-frames 8000 --seed 7

# 80. Record the dashboard on every stage the map hands over (10.66): the surveyed states plus the
#     ones the clearance campaign entered, so a stage that only exists after a clear is included.
#     Per stage it measures the terrain the buffer holds, the distance a held RIGHT+Y walk reaches,
#     how many of six clearance-controller attempts finish the stage, and one kept episode with the
#     10.60 channels. Needs core + ROM, ~10 min per stage. Writes
#     data/raw/smw_dashboard_run_*.npz, results/level_dashboards/<stage>/ and
#     results/level_dashboards_cleared.json.
python -m scripts.record_level_dashboards

# 81. Render the hero and one dashboard per stage, and write the section (10.66). Emulator-free: it
#     reads the recordings. ~2 min per stage for the MP4 plus the derived GIF and still. Writes
#     results/figures/dashboard_run.*, results/figures/dashboard_level_*.*, the per-render metric
#     files and results/level_dashboard_study_metrics.json.
python -m src.evaluation.level_dashboard_study

# 82. Survey the map a cleared stage opened (10.67): the same probe on the world-map state the
#     clearance campaign kept after its clear, instead of booting to one. Needs core + ROM, ~4 min.
#     Writes results/world_map_survey_opened.json; its captures use the `smw_opened_node` prefix and
#     are not committed, because 10.67 compares their bytes rather than recording from them.
python -m scripts.survey_world_map --from-map-state data/raw/smw_world_map_after_round_0_152_136_t2a.state --state-prefix smw_opened_node --output results/world_map_survey_opened.json

# 83. Re-run the published survey into its own artifact (10.67): the power-on route again, every
#     capture hashed, so the README can say it repeats and be checked. Needs core + ROM, ~3 min.
#     Writes results/world_map_survey_replay.json under the discarded `smw_replay_node` prefix.
python -m scripts.survey_world_map --state-prefix smw_replay_node --output results/world_map_survey_replay.json

# 84. Compare the two survey routes and the replay against the published survey (10.67).
#     Emulator-free: it reads the three artifacts and hashes the committed savestates. Writes
#     results/world_map_recheck_metrics.json and the section.
python -m src.evaluation.world_map_recheck_study

# 85. Distribution-free intervals on the 10.56 laws (10.56.1): split conformal prediction at
#     three miscoverage levels, calibrated on the canonical validation split and scored on
#     the test split, against the iid-Gaussian arm and the fit-mean null. Emulator-free,
#     ~10 s. Writes results/conformal_uncertainty_metrics.json and the table README 10.56.1 quotes.
python -m src.evaluation.conformal_uncertainty_benchmark
```

### 11.6 Engineering Workflows (CI, Configs, Parity Baselines, Regression Gates)

```bash
# Canonical install (imports `from src...`) + shortcuts:
pip install -e ".[dev]"
make test        # 340+ unit/smoke/guard tests (emulator tests skip off-Windows)
make test-cov    # with coverage gate (baseline 30%)
make lint        # ruff check src tests scripts
make format      # ruff format src tests scripts
make format-check # ruff format --check (what CI runs)
make typecheck   # mypy on typed core modules
make check-all   # lint + format-check + typecheck + test-cov (the full gate)
make reproduce   # fast CPU smoke benchmark (configs/reproduce.yaml)
make smoke-all   # seconds-scale runs of $12 studies -> results_smoke/ (ignored)
make benchmark sample-efficiency multiseed
make inverse-transfer symbolic-inverse symbolic-tilemap symbolic-engines  # The inverse-problem studies (10.40, 10.43, 10.43.8, 10.43.9)
make inverse-mpc                             # Closed-loop inverse-model comparison (10.44, needs hardware)
smw-pinn check-all # the same gate on Windows, where `make` is usually unavailable
```

* **YAML configs (`configs/*.yaml`):** `benchmark.yaml`, `multiseed.yaml` (K=10 seeds), `sample_efficiency.yaml`, `reproduce.yaml`. Every benchmark accepts `--config`; explicit CLI flags override the file (`src/utils/config.py`).
* **Deterministic loaders:** `create_dataloaders(..., seed=...)` uses an explicit seeded `torch.Generator` + `seed_worker`; episodic splits never duplicate validation episodes into test (`src/environment/dataset_loader.py`).
* **Parameter parity:** `--matched-baseline` adds a compact ~10k-param MLP vs ~10k-param Hard PINN pair (`build_param_matched_mlp`, `MATCHED_HIDDEN_DIMS = [64, 64, 64]`).
* **Per-variable metrics:** `benchmark_metrics.json` now also reports per-channel MSE/MAE/R² (`x, y, vx, vy`) and accuracy/F1 (`c_*`) plus `rollout_multistart` (mean ± std over N starts) — aggregate MSE is dominated by coordinate scale (e.g. 1-epoch smoke: Hard PINN `x`-MSE 0.14 vs MLP 106k).
* **Stronger statistics:** multiseed defaults to K=10 seeds with paired t + Wilcoxon + Cohen's dz, evaluated on single-start *and* multi-start drift.
* **Regression gate:** `tests/test_metrics_regression.py` fails CI if published numbers silently degrade (Hard MSE < 2.0, 0 kinematic violations, N=200 Hard beats N=5000 MLP).
* **CI/Docker:** `.github/workflows/ci.yml` (ruff + mypy + pytest + coverage; uploads `pytest.log` on failure) and `Dockerfile` (CPU base, CUDA via build-arg). Dependabot stays inside the validated torch envelope (`torch<2.7`, `torchvision<0.22`, `numpy<2.1`); widen caps only with hardware re-validation.
* **Refreshed numbers:** `results/benchmark_metrics.json`, `sample_efficiency_metrics.json` and `multiseed_benchmark_metrics.json` were regenerated with deterministic seeded loaders (seed 42) and K=10 seeds; tables in §8.1–§8.4 match those files exactly (`tests/test_metrics_regression.py` enforces it).
* **New frontiers (§10.31–§10.36):** pixel perception (`src/perception/`), hierarchical A\*+MPC (`src/planning/global_planner.py`), reflex ablation + TD-MPC value, connected orphans, spatial-holdout OOD. Emulator-dependent tests skip off-Windows via `@requires_emulator`; Yoshi's Island 2 capture is documented as blocked in §10.36 (with `results/yi2_capture_attempt.json` as the evidence artifact).
* **Reference baselines and audit tooling (§10.37–§10.38):** the zero-parameter engine-rule model (`src/models/analytical_kinematics.py`), the oracle-MPC comparison (`src/evaluation/analytical_baselines.py --hardware`), the closed-loop reproduction audit and the episode-preamble probe (`src/evaluation/mbrl_mpc_benchmark.py --reproduction-check / --preamble-probe`).
* **Asset paths in one place (`src/utils/paths.py`):** the Libretro core, ROM, savestates, datasets and every `results/` output resolve through repo-root-anchored, `SMW_*`-overridable helpers (`require_rom`, `require_core`, `results_file`, `checkpoint_file`), so entry points behave identically from any working directory and a missing ROM produces an acquisition message instead of a traceback. The WRAM register map lives in `src/environment/wram.py`.
* **Cross-platform runner (`src/cli.py`):** `pip install -e .` exposes `smw-pinn`, whose subcommands mirror the Makefile's study targets (`smw-pinn baselines`, `smw-pinn multiseed`, `smw-pinn sindy`, `smw-pinn corrected-physics`) plus a generic `smw-pinn run <module> [args...]`; `tests/test_cli_parity.py` refuses a Makefile study target with no subcommand, which is how ten of them (10.45-10.56) had quietly gone unregistered. The mypy typed-core list lives here, so `make`, CI and `smw-pinn` cannot drift apart.
* **Provenance and artifact index:** every artifact written by these tools embeds a `_meta` block (git SHA and dirty flag, library/CUDA versions, seed, command, UTC timestamp) via `src/utils/provenance.py:write_metrics`, and `results/MANIFEST.md` maps artifact → writer → command → README section. `tests/test_results_manifest.py` fails CI on an ownerless artifact, a fictional writer, a dangling claim, a growing `_meta` exemption list, a checkpoint newer than the result that used it, or a §10 headline that no longer matches its artifact.
* **Fast smoke tests for the §10 studies:** eighteen emulator-free studies that otherwise need minutes or a GPU also ship a seconds-scale config (`configs/smoke_multiseed.yaml`, `smoke_sample_efficiency.yaml`, `smoke_pinn_ensemble.yaml`, `smoke_unified_ppo.yaml`, `smoke_set_multi_entity.yaml`, `smoke_deeponet.yaml`, `smoke_operators.yaml`, `smoke_symbolic_inverse.yaml`, `smoke_symbolic_tilemap.yaml`, `smoke_symbolic_engines.yaml`, `smoke_physics_injection.yaml`, `smoke_metric_decomposition.yaml`, `smoke_plateau_provenance.yaml`, `smoke_projection_cell.yaml`, `smoke_gate_excitation.yaml`, `smoke_effective_velocity.yaml`, `smoke_neural_ode.yaml`, `smoke_sindy.yaml`) plus the seconds-scale §10.54 audit, which needs no config at all, and `make smoke` / `smw-pinn smoke-all` runs them all into `results_smoke/` (git-ignored, so a smoke run can never overwrite a published artifact). `tests/test_smoke_runs.py` executes the two cheapest and contract-checks every config against its entry point's real `--help` output, because a config key the parser does not know is silently ignored. The studies that genuinely need the Libretro core and ROM (the §10.18–§10.34 recordings, the §10.44 closed-loop comparison, the §10.47 grid, the §10.51 projection arms, the §10.53 convention arms and the §10.55 integrator arms flown on the console, and the two excitation recorders of §10.45 and §10.52) are covered by `tests/test_hardware_loops.py` instead.
* **Standardized episode preamble:** closed-loop episodes start with `SnesLibretroEmulator.start_episode()` (restore savestate → force gameplay mode `0x14` → warm-up frames → read state). Before it existed, ~15 scripts copy-pasted three different versions of that preamble and the difference was worth 3.4x progress (§10.38.1).
* **Documentation site (the README without the math budget):** `mkdocs.yml` + `docs/` render this file through mkdocs-material and MathJax, where GitHub's ~1,368-expression rendering limit does not apply - `docs/full-readme.md` is generated from this README by `scripts/sync_docs.py` (never edited by hand) and `.github/workflows/docs.yml` syncs, builds and publishes it on GitHub Pages. Locally: `pip install -r docs/requirements-docs.txt && python scripts/sync_docs.py && mkdocs serve`. The docs toolchain is deliberately outside the validated science envelope that `tests/test_dependency_parity.py` guards, so a site dependency can never constrain the torch/CUDA caps.

---
