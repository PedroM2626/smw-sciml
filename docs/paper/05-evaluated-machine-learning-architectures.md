[← 4](04-mathematical-formulation-of-super-mario-world-physics.md) · [Contents](../../README.md#contents) · [6 →](06-loss-function-formulation-and-the-soft-pinn.md)

## 5. Evaluated Machine Learning Architectures

The state vector at frame $t$ is parameterized as:

$$s_t = \begin{bmatrix} X_t & Y_t & v_{x,t} & v_{y,t} & c_{\text{ground}, t} & c_{\text{ceiling}, t} & c_{\text{left}, t} & c_{\text{right}, t} \end{bmatrix}^T \in \mathbb{R}^8$$
and the commanded controller action vector as:

$$a_t = \begin{bmatrix} a_{\text{jump}} & a_{\text{run}} & a_{\text{up}} & a_{\text{down}} & a_{\text{left}} & a_{\text{right}} \end{bmatrix}^T \in \{0, 1\}^6$$
strictly mapped from SNES joypad button registers: $[B, Y, \text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}]^T$.

The combined input vector is $z_t = [s_t, a_t] \in \mathbb{R}^{14}$. The goal is to predict the next state $\hat s_{t+1} \in \mathbb{R}^8$.

```
+---------------------------------------------------------------------------------------------------+
|                                 TAXONOMY OF EVALUATED ARCHITECTURES                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. STATISTICAL MLP (Black-Box Baseline)                                                          |
|     [s_t, a_t] ---> Dense(128) ---> SiLU ---> Dense(128) ---> SiLU ---> Dense(8) ---> s_hat_{t+1}|
|                                                                                                   |
|  2. STATISTICAL LSTM (Temporal Recurrent Baseline)                                                |
|     [s_t, a_t] ---> LSTM(Hidden=128, Layers=2) ---> Dense(8) -----------------------> s_hat_{t+1}|
|                                                                                                   |
|  3. SOFT-CONSTRAINED PINN (Loss Regularization)                                                   |
|     [s_t, a_t] ---> MLP(128x128) ---> s_hat_{t+1}                                                |
|                            |                                                                      |
|                            v                                                                      |
|                Loss = Loss_data + lambda * Loss_kinematics + lambda * Loss_bounds                 |
|                                                                                                   |
|  4. HARD-CONSTRAINED RESIDUAL PINN (Structural Inductive Bias in Computational Graph)             |
|     [s_t, a_t] ---> MLP_Force(64x64) ---> [delta_vx, delta_vy, c_pred]                          |
|                                                  |                                                |
|                                                  v                                                |
|                         v_hat_{t+1} = clamp(v_t + delta_v, Limits)                                |
|                         X_hat_{t+1} = X_t + v_hat_{x,t+1} / 16.0     <--- EXACT INTEGRATION      |
|                         Y_hat_{t+1} = Y_t + v_hat_{y,t+1} / 16.0     <--- ZERO RESIDUAL          |
+---------------------------------------------------------------------------------------------------+
```

### 5.1 Architecture 1: Statistical MLP (Black-Box Baseline)
- **Topology:** Fully connected feedforward network. Three hidden layers of 128 neurons each (`hidden_dims=[128, 128, 128]`, the constructor default the benchmark trains with), GELU activations with LayerNorm, and a linear output layer producing 8 state variables.
- **Formulation:** $\hat s_{t+1} = \text{MLP}(\theta, z_t)$.
- **Parameters:** 36,744 trainable weights.
- **Nature:** Universal statistical approximator with zero inductive physical priors.

### 5.2 Architecture 2: Statistical LSTM (Temporal Sequence Baseline)
- **Topology:** Two-layer Recurrent LSTM with hidden dimension $h = 128$, followed by a linear projection head mapping to 8 state variables.
- **Formulation:** the trunk carries the hidden state, $(h_t, c_t) = \text{LSTM}(\theta, z_t, (h_{t-1}, c_{t-1}))$,
  and the branch reads it out, $\hat s_{t+1} = W_o h_t + b_o$.
- **Parameters:** 223,368 trainable weights.
- **Objective:** Evaluate whether latent temporal memory can implicitly substitute for explicit physical equations.

### 5.3 Architecture 3: Soft-Constrained PINN (Lagrangian Loss Penalty)
- **Topology:** Identical capacity to the Statistical MLP (three hidden layers of 128 neurons, LayerNorm + GELU): 36,744 parameters, the same as the MLP.
- **Optimization Formulation:** The network predicts all 8 state variables directly, but the training loss penalizes kinematic and boundary violations:

  $$\mathcal L_{\text{total}} = \mathcal L_{\text{data}} + \lambda_{\text{kin}} \mathcal L_{\text{kin}} + \lambda_{\text{bound}} \mathcal L_{\text{bound}} + \lambda_{\text{contact}} \mathcal L_{\text{contact}}$$
- **Objective:** Benchmark the traditional continuous PINN paradigm (Raissi et al., 2019) on stiff discrete dynamics.

### 5.4 Architecture 4: Hard-Constrained Residual PINN (Hard Inductive Bias)
- **Topology:** Compact residual force estimation network ($\text{NN}$, the force head) with hidden layers of 128 neurons each with LayerNorm and GELU activations.
- **Structural Formulation:** Analytical integration is hardcoded directly into the tensor computation graph:
  1. The neural network predicts unmodeled force residuals and contact flags: $[\delta v_{x,t}, \delta v_{y,t}, \hat c_{\text{aux}}] = \text{NN}(z_t)$;
  2. Next-frame velocities are accumulated and clamped to theoretical bounds:

     $$\hat v_{x,t+1} = \text{clamp}(v_{x,t} + \delta v_{x,t}, -72.0, +72.0)$$

     $$\hat v_{y,t+1} = \text{clamp}(v_{y,t} + \delta v_{y,t}, -80.0, +64.0)$$
  3. Discrete Euler integration is applied analytically:

     $$\hat X_{t+1} = X_t + \frac{\hat v_{x,t+1}}{16.0}$$

     $$\hat Y_{t+1} = Y_t + \frac{\hat v_{y,t+1}}{16.0}$$
  4. Contact indicators ($\hat c_{\text{ground}}, \hat c_{\text{ceiling}}, \hat c_{\text{left}}, \hat c_{\text{right}}$) are predicted by the auxiliary collision sub-head.
- **Parameters:** 36,486 trainable weights at the published `hidden_dims=[128, 128, 128]`. The parameter-parity ablation (`--matched-baseline`) uses the compact `[64, 64, 64]` pair instead: 10,054 for this network against 10,184 for the matched MLP.
- **Structural Guarantee:** Discrete kinematic consistency residual ($\hat X_{t+1} - X_t - \hat v_{x,t+1}/16.0$) is **identically zero by computational graph construction**.

---
