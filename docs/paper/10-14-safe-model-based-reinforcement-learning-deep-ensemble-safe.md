[← 10.13](10-13-multi-entity-12d-pinn-world-model-autonomous-hazard.md) · [Contents](../../README.md#contents) · [10.15 →](10-15-comprehensive-hardware-computational-efficiency-profiling.md)

### 10.14 Safe Model-Based Reinforcement Learning (Deep Ensemble Safe MBPO)

To address model exploitation in active online learning, we integrated the Deep Ensemble of 5 Hard PINNs (`src/models/pinn_ensemble.py`) into the active closed-loop engine (`src/training/online_mbpo.py --safe`).

#### Algorithmic Safeguards:
1. **Epistemic Disagreement:** At every step of imaginary rollout in `PINNVectorEnv`, the epistemic uncertainty $\sigma(s, a)$ is computed across the ensemble members.
2. **Adaptive Rollout Truncation:** If $\sigma(s, a) \gt \tau_{\text{epistemic}} = 1.2$, the trajectory is truncated immediately, halting imaginary rollouts before ungrounded transitions distort the policy.
3. **Pessimistic Reward Regularization:**

   $$r_{\text{safe}}(s, a) = r(s, a) - \beta \cdot \sigma(s, a) \quad (\beta = 0.5)$$

#### Empirical Online Safe MBPO Results:
| Safe MBPO Iteration | Real Console Frames Collected | Buffer Size ($\mathcal D_{\text{env}}$) | Imagined Throughput (GPU) | Real Console Progress |
| :---: | :---: | :---: | :---: | :---: |
| **Iteration 1** | 500 frames | 2,500 | 65,100 FPS | +0.0 px |
| **Iteration 2** | 500 frames | 3,000 | 79,218 FPS | **+39.8 px** |

![Safe MBPO Convergence](../../results/figures/online_mbpo_safe_convergence.png)

---
