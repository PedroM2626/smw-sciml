[← 10.3](10-03-analysis-of-lstm-performance-and-the-markovian.md) · [Contents](../../README.md#contents) · [10.5 →](10-05-implications-for-model-based-reinforcement-learning-mbrl.md)

### 10.4 Long-Horizon Rollout Dispersion (Frame 120 Drift)
In open-loop rollout evaluations (120 frames / 2 seconds without ground-truth environmental feedback), the Hard PINN preserved **0% kinematic violations**, yet its open-loop trajectory was 78.13 px away from ground truth at frame 120 (288.73 px averaged over the 10-start multi-start variant):
* **Subpixel Sensitivity in Non-Differentiable Collisions:** Platformer engines resolve collisions via discrete axis-aligned bounding boxes (*AABB Hitboxes*). An infinitesimal prediction deviation of 0.1 pixel at apex descent determines whether Mario lands on a pipe or slips past its edge.
* **Bifurcation of Discrete Events:** Once an open-loop landing event diverges, subsequent trajectories diverge geometrically. The model's kinematics remain analytically flawless at every step, but the trajectory branches away from ground truth. This highlights the need for hybrid Bayesian or probabilistic contact heads in ultra-long horizons.

---
