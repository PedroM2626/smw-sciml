[← 10.11](10-11-spatial-translation-invariant-pinn-dynamics.md) · [Contents](../../README.md#contents) · [10.13 →](10-13-multi-entity-12d-pinn-world-model-autonomous-hazard.md)

### 10.12 Closed-Loop Active Model-Based Policy Optimization (Online MBPO)

Closing the loop between offline modeling and online reinforcement learning, we implemented **Online MBPO** (`src/training/online_mbpo.py`):
1. **Real Data Aggregation:** Gathers authentic transitions from the SNES emulator core into an active replay buffer $\mathcal D_{\text{env}}$.
2. **Continual PINN Adaptation:** Periodically fine-tunes the Hard Residual PINN on newly discovered state regions.
3. **Branched Model Rollouts:** Samples states $s \sim \mathcal D_{\text{env}}$ and simulates short branched trajectories ($k = 10$ steps) within the in-GPU vectorized PINN simulator, preventing compounding trajectory drift.
4. **Policy Optimization:** Trains the Actor-Critic policy using PPO across 40,000 imagined transitions per iteration in under 2.3 seconds (>31,000 FPS).

#### Online Iterative Convergence Results:
| MBPO Iteration | Real Console Transitions Collected | Buffer Size ($\mathcal D_{\text{env}}$) | Imagined Training Throughput | Policy Return | Real Console Max Progress |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Iteration 1** | 1,000 frames | 3,000 | 27,862 FPS | +11.40 | +114.4 px |
| **Iteration 2** | 1,000 frames | 4,000 | 31,661 FPS | +12.73 | +111.1 px |
| **Iteration 3** | 1,000 frames | 5,000 | 28,059 FPS | **+14.18** | **+115.5 px** |

![Online MBPO Convergence](../../results/figures/online_mbpo_convergence.png)

---
