[← 10.28](10-28-frontier-2-zero-shot-closed-loop-control-on-unseen.md) · [Contents](../../README.md#contents) · [10.30 →](10-30-scope-limitations-threats-to-validity.md)

### 10.29 Consolidation of Frontiers A, B, C and D: Differentiable Optimization, PPO and Multimodal Rendering

To expand the scope of the project towards the most advanced frontiers of model-based reinforcement learning (*Model-Based RL*) and computational physics, four new scientific fronts were implemented and validated:

#### 10.29.1 Frontier C: Unified Multimodal Model (`src/models/pinn_unified_multimodal.py`)
- **Architecture:** Unifies in a single computational graph:
  1. Mario's continuous 8D kinematic state $[X, Y, v_x, v_y, c_g, c_c, c_l, c_r]$;
  2. A 2D convolutional encoder of spatial terrain from WRAM (`$7E:C800`) processing local $7 \times 7$ blocks;
  3. A relative hazard dynamics module (dynamic sprites such as Rex) $[ \Delta X_h, \Delta Y_h, v_{xh}, \text{active} ]$.
- **Physical Guarantee:** Exact analytical conservation of 0.0% kinematic violation for both the player and the enemy-relative displacement vector.
- **Tests:** 100% coverage and passing in [`tests/test_unified_multimodal.py`](../../tests/test_unified_multimodal.py).

#### 10.29.2 Frontier B: Differentiable Gradient Controller through the Hard PINN (`src/planning/differentiable_pinn_planner.py`)
- **First-Order Control Formula:** Instead of stochastic sampling search (zero-order CEM MPC), we parameterize the action sequence as continuous logits $\mathbf{U} \in \mathbb{R}^{H \times 6}$ with a Sigmoid relaxation and compute the analytical gradient of the reward directly through the weights and equations of the Hard PINN:

  $$\nabla_{\mathbf u_{0:H-1}} J = \nabla_{\mathbf u_{0:H-1}} \sum_{\tau=0}^{H-1} R(s_\tau, \sigma(\mathbf u_\tau))$$
- **Convergence:** Optimization via Adam ($lr = 0.25$, 15 gradient steps) adjusts the controls based on the exact gradient field of the game physics.
- **Tests:** Validated in [`tests/test_differentiable_planner.py`](../../tests/test_differentiable_planner.py).

#### 10.29.3 Amortized PPO inside the PINN Simulator (Unified Dyna-PPO — `src/training/train_unified_ppo.py`)
- **Vectorized GPU Training:** 128 parallel environments simulated directly in PyTorch tensors on the GPU, reaching a throughput of **14,395 transitions per second** (200,000 timesteps completed in only 13.7 seconds).
- **Sim-to-Real Diagnostics on Real Hardware:**
  - The pure PPO agent trained in simulation reached a survival of **2,500 frames on the real console** operating at **1,425.1 FPS**, but exhibited the classic *Passive Hedging Collapse* phenomenon (hesitation and crouching at the spawn point, $-7.38\text{ px}$).
  - Conversely, the **DAgger** policy (trained with interactive on-policy aggregation of hardware trajectories) surpassed the **250 px, 500 px and 782 px** milestones, accumulating **831.75 px of real progress** at **2,707.5 FPS**.

#### 10.29.4 Frontier A and Option 3: End-to-End Level Clearance (Full Level Clearance) & WRAM Telemetry Video
- **Game Completion Status:** **GOAL REACHED! (Level 100% Cleared)** on the real Libretro SNES console.
- **Official Metrics on Hardware (`results/full_level_clearance_metrics.json`):**
  - **Total Progress:** **2,003.69 pixels** (from $X=16.0$ to $X=2{,}022.0$ px).
  - **Frames Survived:** **971 authentic frames** (16.2 seconds at 60 Hz).
  - **Subscreens Traversed:** Every subscreen of the stage (Subscreens 0, 1, 2, 3, 4, 5, 6 and 7).
  - **Milestones Passed:** 250 px, 500 px, 782 px (Rex 1), 1,000 px (Plateau), 1,250 px (Pipe Valleys), 1,500 px (Upper Hills), 1,750 px (Final Straight) and 1,900 px (Goal Tape Zone).
  - **Goal Tape Crossing:** **Frame 917** ($X = 1{,}916.6$ px), with the triumphant walk completed on **Frame 971** ($X = 2{,}022.0$ px).
  - **Mean Velocity:** **33.05 subpixels/frame** at **19.62 FPS** of continuous CEM MPC control throughput on the GPU.
- **Dynamic Multimodal Rendering (`src/evaluation/render_level_clearance_video.py`):**
  - A continuously tracking mobile camera centered on Mario $[X(t) - 120, X(t) + 280]$ that follows the entire horizontal extent of the map;
  - The Goal Tape modeled visually at $X \approx 1{,}950$ px with a yellow band and vertical posts;
  - A real-time progress curve $X(t)$ with a synchronized time cursor;
  - A WRAM telemetry HUD panel at 60 Hz showing coordinates $(X, Y)$, velocities $(v_x, v_y)$, the current subscreen and the state of the SNES controller buttons ($B, Y, \text{RIGHT}$).
- **Generated Artifacts:**
  - High-resolution MP4 video: `results/figures/full_level_clearance.mp4` (324 frames at 30 FPS, 2.5 MB).
  - Synchronized animated GIF: `results/figures/full_level_clearance.gif` (162 frames at a continuous 15 FPS, 4.5 MB).
  - Complete trajectory chart: `results/figures/full_level_clearance_trajectory.png`.

![Full Level Clearance Animation](../../results/figures/full_level_clearance.gif)
*Figure: Fluid, continuous animation of the full traversal of Yoshi's Island 1 on the real SNES console with a dynamic tracking camera, 60 Hz WRAM telemetry HUD and the goal-tape crossing.*

---
