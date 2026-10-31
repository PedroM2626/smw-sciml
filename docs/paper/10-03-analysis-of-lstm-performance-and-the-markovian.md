[← 10.2](10-02-the-soft-pinn-fallacy-in-discrete-dynamical.md) · [Contents](../../README.md#contents) · [10.4 →](10-04-long-horizon-rollout-dispersion-frame-120-drift.md)

### 10.3 Analysis of LSTM Performance and the Markovian Hypothesis
The recurrent LSTM achieved the poorest test loss (39.2194) and early-stopped at epoch 11. The theoretical justification is clear:
* **Full Observability:** Because the WRAM state $s_t$ already exposes canonical kinematic coordinates ($X_t, Y_t, v_{x,t}, v_{y,t}$ and contact flags), state transitions obey the **first-order Markov property**:

  $$\mathbb{P}(s_{t+1} \mid s_t, a_t, s_{t-1}, \dots, s_0) = \mathbb{P}(s_{t+1} \mid s_t, a_t)$$
* **Overparameterization and Instability:** Introducing recurrent cells with 223,368 parameters forced the model to infer spurious historical dependencies, generating optimization instability and local minima. In fully observable environments, explicit Markovian inductive models decisively outperform recurrent memory.

---
