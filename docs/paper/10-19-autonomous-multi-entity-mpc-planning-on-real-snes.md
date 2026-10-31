[← 10.18](10-18-genuine-multi-entity-dataset-supervised-hazard-dynamics-training.md) · [Contents](../../README.md#contents) · [10.20 →](10-20-spatial-discrete-tilemap-perception-via-wram-7e.md)

### 10.19 Autonomous Multi-Entity MPC Planning on Real SNES Console (782 px Rex Evasion)

Equipped with the supervised `MultiEntityPINNDynamics` model, we extended the Model Predictive Controller (`src/planning/mpc_planner.py`) with dynamic hazard-aware trajectory optimization:
- **Hitbox Collision Avoidance:** Penalizes candidate paths where $|\Delta X_h| \lt 14.0\text{ px}$ and $-10.0\text{ px} \lt \Delta Y_h \lt 16.0\text{ px}$ (`hazard_penalty = 600.0`).
- **Apex Evasive Vault Reward:** Awards `leap_bonus = 200.0` when the simulated trajectory initiates an airborne leap that passes the enemy horizontally ($X_{\text{mario}} \gt X_{\text{hazard}}$).
- **Closed-Loop Live Hardware Benchmark (`src/evaluation/evaluate_multi_entity_mpc.py`):**
  - **Survival:** 400 / 400 frames (100% survival rate).
  - **Total Progress:** **782.94 pixels** - 2.4x the best hand-coded controller in the table below (328.9 px) and 6.8x the blind 8D policy's 115.0 px, which is the first autonomous Rex evasion on the console.
  - **Rex Evasion:** **100% Autonomous** via CEM trajectory optimization (horizon $H=16$, candidates $N=256$, planning throughput 23.7 FPS).
  - **Zero Heuristics:** No hand-crafted `if/else` triggers; jump timing and dash momentum are planned strictly through forward dynamics simulation.

| Metric | Blind 8D Policy | Heuristic Rule Controller | Multi-Entity 12D Policy | **Autonomous Multi-Entity MPC (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **Rex Evaded** | False | True (Hand-coded) | False | **True (Autonomous CEM)** |
| **Total Progress** | 115.0 px | 328.9 px | -7.4 px | **782.94 px** |
| **Survived Frames** | 173 | 500 | 413 | **400 / 400 (Full Trial)** |
| **Mean Velocity ($v_x$)** | 19.5 | 10.5 | N/A | **31.41 subpx/frame** |

![Multi-Entity MPC Trajectory](../../results/figures/multi_entity_mpc_trajectory.png)

---
