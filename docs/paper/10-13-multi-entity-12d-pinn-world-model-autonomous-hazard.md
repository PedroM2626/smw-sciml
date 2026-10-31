[← 10.12](10-12-closed-loop-active-model-based-policy-optimization-online-mbpo.md) · [Contents](../../README.md#contents) · [10.14 →](10-14-safe-model-based-reinforcement-learning-deep-ensemble-safe.md)

### 10.13 Multi-Entity 12D PINN World Model & Autonomous Hazard Evasion

To generalize world modeling beyond a single kinematic agent, we designed the **Multi-Entity PINN** (`src/models/pinn_multi_entity.py`). This architecture models Mario (8D) simultaneously with dynamic stage hazards (4D: $\Delta X_{\text{hazard}}, \Delta Y_{\text{hazard}}, v_{x,\text{hazard}}, \text{active}$), yielding a **12-Dimensional Joint State Representation**.

#### Analytical Relative Kinematic Consistency:
Rather than delegating the relative motion of entities to black-box regression, the relative kinematic displacement is embedded directly into the PyTorch computational graph:

$$\hat X_{\text{mario}, t+1} = X_{\text{mario}, t} + \frac{\hat v_{x, \text{mario}, t+1}}{16.0}$$

$$\Delta\hat X_{\text{hazard}, t+1} = \Delta X_{\text{hazard}, t} + \frac{\hat v_{x, \text{hazard}, t+1} - \hat v_{x, \text{mario}, t+1}}{16.0}$$

This guarantees exact spatial consistency of inter-entity relative displacements with **0.0% analytical violation**.

#### End-to-End Autonomous Policy Optimization:
Using `src/training/dyna_ppo_sprites.py`, an Actor-Critic policy was trained entirely inside the GPU-vectorized PINN simulation (`PINNVectorEnv` at >16,000 FPS). By incorporating hazard collision penalties (-80.0) and forward leap milestone rewards (+45.0), the policy autonomously discovers the coordinated jump timing required to leap over approaching Rex hazards without any manual trigger-edge heuristics.

| Training Metric | Value (300k Timesteps) |
| :--- | :---: |
| **Peak Policy Return** | **+164.09** |
| **Hazard Collision Reduction** | **333 down to 207 collisions/update (-37.8%)** |
| **Vectorized In-GPU Throughput** | **17,644 transitions/sec** |

![Multi-Entity Dyna-PPO Learning Curve](../../results/figures/dyna_ppo_multi_entity_curve.png)

---
