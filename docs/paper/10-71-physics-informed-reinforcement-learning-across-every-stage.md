[← 10.70](10-70-what-the-way-across-0x27-requires-and.md) · [Contents](../../README.md#contents) · [11 →](11-complete-reproducibility-guide.md)

### 10.71 Physics-Informed Reinforcement Learning Across Every Playable Stage (`PIML-PPO`), and the Full-Stage Dashboards

10.69 and 10.70 established that stages `0x29` (*Yoshi's Island 2*) and `0x27` (*Yoshi's Island 3*) are both solvable under the simple potential-maximum reward (`r_t = (x_t - x^*_{t-1}) / 16` on a new maximum of `x`, plus `+10` on an engine-verified clear), using a genetic algorithm on `0x29` and a Go-Explore cell archive on `0x27`. However, two scientific and engineering gaps remained:

1. **Neither stage was cleared by a Physics-Informed Reinforcement Learning (SciML + RL) policy.** While `0x2A` (*Yoshi's Island 1*) used the `MultiEntityPINN` world model inside an MPC planner (10.66) and 10.39 introduced Physics-Informed Model-Free RL (`PIML-MFRL`) on the 8D flat-stage benchmark, `0x29` and `0x27` had no closed-loop RL policy trained with physics-informed inductive biases that could react to the 16-bit Map16 tile geometry (`7x7` window) and active sprite slots (`3x4` channels) while providing actor probabilities $\pi_\theta(a \mid s_t)$ and critic values $V_\phi(s_t)$ on every frame.
2. **The dashboard videos (`dashboard_level_3.mp4`, `dashboard_level_4.mp4`, and `all_covered_stages.mp4`) still showed the pre-10.69 failure runs.** Because `scripts/record_level_dashboards.py` originally flew `terrain-runner` on `0x29` and `0x27`, the rendered Level 3 and Level 4 dashboards ended in pit deaths (`410.2 px` and `506.0 px`) instead of full stage conclusions.

`src/training/piml_stage_ppo.py` closes both gaps by training a closed-loop **Physics-Informed Proximal Policy Optimization (`PIML-PPO`)** Actor-Critic agent on `0x29` and `0x27`, and `scripts/record_level_dashboards.py` flies the trained `piml-ppo` policy to record and render the full-clear dashboards across all four sessions.

#### 10.71.1 Architecture and Physics-Informed RL Objective

`PIMLStageActorCritic` maps an 81-dimensional Markov observation vector $o_t \in \mathbb{R}^{81}$ paired with the raw 8-dimensional physical state $p_t = [x_t, y_t, v_{x,t}, v_{y,t}, c_{\text{gnd}}, c_{\text{ceil}}, c_{\text{left}}, c_{\text{right}}]^\top \in \mathbb{R}^8$ to macro-action probabilities (`N_ACTIONS = 8`, held for `FRAME_SKIP = 4` frames), a scalar state value $V_\phi(o_t, p_t)$, and a physics-constrained 4-frame kinematic prediction $\hat p_{t+1}$.

To ensure the policy is strictly Markovian in closed loop without relying on open-loop step indices, $o_t$ augments the 69-dimensional spatial perception vector of 10.69 (`8D` normalised pose/velocity/contacts + `7x7 = 49` measured 16-bit tile solidity/hazard encodings + `3x4 = 12` nearest active sprite slots) with a 12-dimensional internal-state summary:
- **Progress and stall state (`4D`):** normalised maximum progress $x^*_t / 5200$, instantaneous progress gap $(x^*_t - x_t) / 160$, consecutive stall counter $\min(\tau_{\text{stall}}, 50)/50$, and the active `PlayerHold` message-box latch indicator, plus two sinusoidal phase pairs $(\sin(x_t / 32), \cos(x_t / 32))$ and $(\sin(y_t / 32), \cos(y_t / 32))$ that resolve sub-tile platform alignment across identical $7\times 7$ tile patches.
- **Previous macro-action one-hot (`8D`):** $e_{a_{t-1}} \in \{0, 1\}^8$, making velocity-hysteresis and jump-hold continuation states distinguishable when two rollout frames visit the same $(x_t, y_t, v_{x,t}, v_{y,t})$ coordinate.

Four SciML mechanisms constrain the Actor, Critic, and representation simultaneously:

1. **Discrete Control Barrier Function Logit Projection (`DiscreteCBFCategoricalFilter`, Approach B of 10.39):** Before forming the categorical distribution $\pi_\theta(\cdot \mid o_t)$, the raw policy logits $z_t \in \mathbb{R}^8$ pass through the differentiable safety barrier filter (`src/models/cbf_projection.py`, $\beta_{\text{cbf}} = 0.25$), which projects out unsafe actions when $p_t$ approaches contact/hazard boundaries while preserving gradient flow via the straight-through estimator.
2. **Hamilton-Jacobi-Bellman / Control-Lyapunov Critic Regularisation (`PhysicsInformedCriticLoss`, Approach A of 10.39):** Differentiating $V_\phi(o_t, p_t)$ with respect to the continuous physical coordinates $p_t$ via `torch.autograd.grad(..., create_graph=True)`, the critic is penalised whenever the Lie derivative along the empirical 4-frame state increment $\dot p_t = (p_{t+1} - p_t)/4$ violates exponential Lyapunov decay on danger states:

$$\mathcal L_{\text{HJB}}(\phi) = \mathbb E_t \left[ w_{\text{danger}}(p_t) \, \mathrm{ReLU}\!\left( \nabla_{p_t} V_\phi(o_t, p_t) \cdot \dot p_t + \alpha \, V_\phi(o_t, p_t) \right)^2 \right]$$

3. **Expected Action Physics Violation (`ActionPhysicsViolation`, Approach C of 10.39):** Penalises the policy probabilities $\pi_\theta(a \mid o_t)$ against ground non-penetration and horizontal speed-envelope violations (`src/losses/physics_rl_losses.py`).
4. **Hard-Constrained Residual PINN Auxiliary Dynamics Head (`HardKinematicsHead`):** A physics-informed world-model head attached to the shared trunk predicts the 4-frame velocity residual $(\Delta \hat v_{x,t}, \Delta \hat v_{y,t})$ and integrates position through the exact fixed-point kinematic identity of Super Mario World (`frame_skip = 4`, sub-pixel divisor `16.0`):

$$\hat v_{x,t+1} = \mathrm{clamp}(v_{x,t} + \Delta \hat v_{x,t}, -72, 72), \qquad \hat x_{t+1} = x_t + \frac{4}{16}\hat v_{x,t+1}, \qquad \hat y_{t+1} = y_t + \frac{4}{16}\hat v_{y,t+1}$$

so the discrete kinematic consistency identity $16 (\hat x_{t+1} - x_t) - 4 \hat v_{x,t+1} = 0$ is satisfied to floating-point precision by construction on every state.

#### 10.71.2 Go-Explore Phase-2 Frontier Resets, Self-Imitation, and FP32 Matmul Determinism

Training a neural policy on long-horizon platforming stages with deceptive traps (`0x27`'s first-screen lower-floor trap at $x = 544\text{ px}$ and second vertical barrier at $x = 3768\text{ px}$) requires combining on-policy exploration with frontier state resets (Go-Explore Phase 2; Ecoffet et al., *Nature* 2021) and Self-Imitation Learning (Oh et al., *ICML* 2018):

- **Frontier savestate resets + on-policy PPO rollouts:** Each training epoch collects `64` on-policy macro-steps (`256` emulator frames) initialized from emulator savestates sampled along the stage's progression frontier (`8` evenly spaced checkpoints along the best high-return trajectory), computing GAE($\gamma = 0.99, \lambda = 0.95$) advantages and updating the joint objective:

$$\mathcal L_{\text{total}} = \mathcal L_{\text{PPO-Clip}} + c_v \mathcal L_{\text{SmoothL1}}(V_\phi, \hat R) - c_e \mathcal H(\pi_\theta) + \lambda_{\text{HJB}} \mathcal L_{\text{HJB}} + \lambda_{\text{act}} \mathcal L_{\text{ActPhys}} + \lambda_{\text{PINN}} \mathcal L_{\text{PINN}} + \lambda_{\text{SIL}} \mathcal L_{\text{SIL}}$$

- **Batch-vs-Single-Step Floating-Point Determinism on Ada Lovelace GPUs:** During training, batched evaluation feeds `[T, 81]` tensors through CUDA `GEMM` kernels, which on Ampere/Ada GPUs default to `TF32` (10-bit mantissa), whereas closed-loop frame-by-frame control feeds `[1, 81]` vectors through `GEMV` kernels in strict `FP32` (23-bit mantissa). When two macro-actions have nearly tied logits, a `1e-4` `TF32`-vs-`FP32` rounding shift flips `argmax` at a single jump frame and diverges the chaos-sensitive SNES trajectory. `src/training/piml_stage_ppo.py` disables `TF32` matrix multiplications (`torch.backends.cuda.matmul.allow_tf32 = False`) and enforces a minimum logit margin (`min_margin >= 0.5`) across all trajectory states before certifying a checkpoint.

#### 10.71.3 Closed-Loop SciML + RL Results on `0x29` and `0x27`

| stage | savestate | closed-loop policy | progress to goal tape | episode length | return ($R$) | Hard PINN pos MAE | kinematic residual $\max \|16\Delta\hat x - 4\hat v'_x\|$ | engine verdict |
|---|---|---|---|---|---|---|---|---|
| `0x29` (*Yoshi's Island 2*) | `smw_map_node_68_135_t29.state` | `PIML-PPO` (`piml_ppo_smw_map_node_68_135_t29.pt`) | **4808.0625 px** (`max_x = 4824.0625 px`) | 3288 frames (822 macro-steps) | 310.5039 | 2.0676 px | 0.0039 | **cleared** (`0x14 -> 0x0c -> 0x0d -> 0x0e`, lives `4 -> 4`) |
| `0x27` (*Yoshi's Island 3*) | `smw_map_round_1_152_104_t27.state` | `PIML-PPO` (`piml_ppo_smw_map_round_1_152_104_t27.pt`) | **5065.6250 px** (`max_x = 5081.6250 px`) | 2252 frames (563 macro-steps) | 326.6016 | 1.2649 px | 0.0039 | **cleared** (`0x14 -> 0x0c -> 0x0d -> 0x0e`, lives `4 -> 4`) |

Both trained `PIML-PPO` checkpoints achieve **100% closed-loop stage completion** from `t = 0` in a fresh emulator process, matching the exact goal-tape coordinates of 10.69 and 10.70 while operating as closed-loop neural policies with live state-value estimates ($V_\phi(s_0) = 41.20 \to V_\phi(s_T) = 10.31$ on `0x29`; $V_\phi(s_0) = 46.77 \to V_\phi(s_T) = 11.03$ on `0x27`) and a `HardKinematicsHead` accuracy of `2.07 px` (`0x29`) and `1.26 px` (`0x27`) over 4-frame macro-steps.

#### 10.71.4 Probing the 10 Newly Encountered Mid-Stage Sprite IDs and Rendering All 4 Concluded Dashboards

Flying `piml-ppo` to the goal tape of `0x29` and `0x27` inside `scripts/record_level_dashboards.py` brings the dashboard recorder past the early-stage pits where 10.68's `terrain-runner` had died, exposing **10 mid- and late-stage sprite IDs** that no previous dashboard recording had reached:
- **Moving and rotating platforms on `0x27` (`platform` class):** `0x55` (checkerboard platform, horizontal), `0x57` (checkerboard platform, vertical), `0x59` (turn-block bridge, horizontal/vertical), and `0x5A` (turn-block bridge, horizontal). Under the counterfactual drop/walk probe of `scripts/probe_sprite_behaviors.py`, all four arrest Mario's fall (`vy -> 0`) above the floor without taking a life or firing the hurt timer, expanding the `platform` class of `src/environment/sprite_behavior_table.py` from `0` to `4` empirically measured IDs (`10 stompable`, `15 neutral`, `4 interact`, `4 platform` across `33` probed IDs).
- **Mid-stage enemies, power-ups, and collectibles:** `0x09` (green bouncing Koopa, `stompable`), `0x0B` (red vertical bouncing Koopa, `stompable`), `0x74` (Super Mushroom, `interact`), `0x8E` (Warp Hole, `neutral`), `0x95` (Coin Cloud, `neutral`), and `0xC7` (Invisible Mushroom, `interact`).

With `piml-ppo` wired into `scripts/record_dashboard_run.py` and `scripts/record_level_dashboards.py`, all four world-map sessions are now recorded and rendered to completion in `results/figures/dashboard_level_1.mp4` through `dashboard_level_4.mp4` and the combined four-stage reel `results/figures/all_covered_stages.mp4`:

| dashboard video | stage (`translevel`) | controller / policy | recorded frames (`engine_exit_frame`) | max progress (`px`) | lives (`start -> end`) | engine conclusion |
|---|---|---|---|---|---|---|
| `results/figures/dashboard_level_1.mp4` | `0x2A` (*Yoshi's Island 1*) | `mpc` (`MultiEntityPINN`, seed 7) | 7114 (`6794`) | 5010.5625 px | `4 -> 5` | `engine_left_0x14_to_0x0C` (**cleared**) |
| `results/figures/dashboard_level_2.mp4` | `0x28` (*Yoshi's House*, non-stage hub) | `terrain-runner` (`LEFT`) | 462 (`142`) | 0.0000 px | `4 -> 4` | `engine_left_0x14_to_0x0B` (**exited to map**) |
| `results/figures/dashboard_level_3.mp4` | `0x29` (*Yoshi's Island 2*) | `piml-ppo` (`PIMLStageActorCritic`, seed 42) | 4207 (`3887`) | 5008.5625 px | `4 -> 4` | `engine_left_0x14_to_0x0C` (**cleared**) |
| `results/figures/dashboard_level_4.mp4` | `0x27` (*Yoshi's Island 3*) | `piml-ppo` (`PIMLStageActorCritic`, seed 42) | 3172 (`2852`) | 5267.0625 px | `4 -> 4` | `engine_left_0x14_to_0x0C` (**cleared**) |

Regenerate: `make piml-stage-ppo` (or `smw-pinn piml-stage-ppo`; needs core + ROM), then `make record-level-dashboards`, `make probe-sprites`, `make level-dashboard-study`, and `make perception-correction-study`. Artifacts: `results/piml_stage_ppo_metrics.json`, `results/checkpoints/piml_ppo_smw_map_node_68_135_t29.pt`, `results/checkpoints/piml_ppo_smw_map_round_1_152_104_t27.pt`, `results/level_dashboard_study_metrics.json`, `results/figures/dashboard_level_3.mp4`, `results/figures/dashboard_level_4.mp4`, and `results/figures/all_covered_stages.mp4`.
