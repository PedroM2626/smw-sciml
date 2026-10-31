[← 10.26](10-26-comprehensive-ablation-study-clamping-horizon-drift-cem.md) · [Contents](../../README.md#contents) · [10.28 →](10-28-frontier-2-zero-shot-closed-loop-control-on-unseen.md)

### 10.27 Master Algorithm Comparison Table (World Models, MPC & Reactive Policies)

> [!NOTE]
> **Why do isolated predictive models have no direct progress metric?**
> A world model $s_{t+1} = f_\theta(s_t, a_t)$ is strictly a dynamic transition function that maps state and action to the next state; it **is not a control policy** $\pi(a_t | s_t)$. To generate commands in real time on the SNES console, a world model must be coupled to a trajectory optimizer (such as Model Predictive Control — CEM/Random Shooting) or distilled into a reactive neural network. Below we report the performance of both the isolated transition function and the complete closed-loop system on real hardware.

The table summarizes the totality of the empirical experiments conducted with genuine WRAM telemetry from *Super Mario World*:

| Category | Algorithm / Model | Test MSE (Single-Step) | Kinematic Violation (%) | Sim-to-Real Tracking Error | Real SNES Stage A Prog. (px) | Real SNES Stage B (Yoshi's House) | Throughput / Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Predictive Models (Isolated Dynamics)** | Statistical MLP | 16.4717 | 98.3% | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 1,349,125 FPS |
| | Statistical LSTM | 39.2194 | 100.0% | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 219,827 FPS |
| | Soft-Constrained PINN | 53.8167 | 100.0% | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 1,325,302 FPS |
| | **Hard Residual PINN (Ours)** | **0.5783** | **0.0%** | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 675,683 FPS |
| | Translation-Invariant PINN | 12.5010 (OOD) | **0.0%** | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 650,000 FPS |
| | Deep Ensemble (E=5) | 0.5120 | **0.0%** | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 141,430 FPS |
| | Tilemap-PINN (7x7 WRAM) | 51.4192 (98.7% Acc) | **0.0%** | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 450,000 FPS |
| | Set-Multi-Entity PINN | Exact 0.0% Rel. | **0.0%** | — | *(Requires MPC/Actor)* | *(Requires MPC/Actor)* | 320,000 FPS |
| **Closed-Loop Control (MPC + World Model)** | Random Actions Baseline | — | — | — | 241.50 px (300f) | 183.40 ± 35.71 px (5 seeds) | 50,124 FPS |
| | MPC + Statistical MLP | — | 100.0% | 90.26 px | 66.13 px (300f) | 44.38 ± 22.01 px (5 seeds) | 72.0 FPS |
| | MPC + Soft-Constrained PINN | — | 100.0% | 219.25 px | 38.69 px (300f) | 88.05 ± 23.63 px (5 seeds) | 73.7 FPS |
| | **MPC + Hard Residual PINN (Ours)** | — | **0.0%** | **0.28 px** | **571.75 px (300f)** | **522.88 ± 335.33 px (5 seeds)** | 53.2 FPS |
| | **Hazard-Aware MPC 12D (Ours)** | — | **0.0%** | **4.12 px** | **782.94 px (400f)** | — | 23.7 FPS |
| | **Extended Navigation MPC (Ours)** | — | **0.0%** | **3.85 px** | **1,016.06 px (627f)**| — | 25.3 FPS |
| | **Full Level Clearance MPC (Ours)** | — | **0.0%** | **3.85 px** | **2,003.69 px (971f - GOAL CLEARED)** | — | 19.6 FPS |
| **Amortized Policies (Reactive Neural Networks)** | Model-Free PPO (Direct SNES) | — | — | — | 210.50 px (350f) | — | ~60 FPS |
| | Dyna-PPO 8D (Simulator) | — | — | — | 115.00 px (173f) | — | ~500 FPS |
| | Dyna-PPO 12D Multi-Entity | — | — | — | -7.38 px (Collapse) | — | ~500 FPS |
| | Distilled Policy (1-step BC) | 0.1740 BCE | **0.0%** | — | 115.00 px (174f) | — | **2,900.9 FPS** |
| | **DAgger Policy (3-iter - Ours)** | **0.1671 BCE** | **0.0%** | — | **831.75 px (500f)** | 830.50 ± 0.00 px (5 seeds) | **3,064.9 FPS** |

---
