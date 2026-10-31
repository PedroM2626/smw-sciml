[← 5](05-evaluated-machine-learning-architectures.md) · [Contents](../../README.md#contents) · [7 →](07-experimental-protocol-data-partitioning-hyperparameters.md)

## 6. Loss Function Formulation and the Soft PINN Dilemma

### 6.1 Supervised Data Loss ($\mathcal L_{\text{data}}$)
We employ Smooth L1 loss (Huber Loss) with $\delta = 1.0$ for robust regression against boundary transition outliers:

$$\mathcal L_{\text{data}}(\theta) = \frac{1}{B} \sum_{i=1}^B \mathcal H_\delta (\hat s_{t+1}^{(i)} - s_{t+1}^{(i)}), \quad \mathcal H_\delta(u) = \begin{cases} 0.5 u^2 & \text{if } |u| \lt \delta \\ \delta(|u| - 0.5\delta) & \text{otherwise} \end{cases}$$

### 6.2 Eulerian Kinematic Loss ($\mathcal L_{\text{kin}}$)
Measures the squared deviation between spatial displacement and physical velocity scaled by 16:

$$\mathcal L_{\text{kin}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ \left( \hat X_{t+1}^{(i)} - X_t^{(i)} - \frac{\hat v_{x,t+1}^{(i)}}{16.0} \right)^2 + \left( \hat Y_{t+1}^{(i)} - Y_t^{(i)} - \frac{\hat v_{y,t+1}^{(i)}}{16.0} \right)^2 \right]$$

### 6.3 Operational Boundary Violation Loss ($\mathcal L_{\text{bound}}$)
Penalizes velocities exceeding engine terminal limits:

$$\mathcal L_{\text{bound}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ \max(0, |\hat v_{x,t+1}^{(i)}| - 72.0)^2 + \max(0, \hat v_{y,t+1}^{(i)} - 64.0)^2 \right]$$

### 6.4 Ground Contact Consistency Loss ($\mathcal L_{\text{contact}}$)
Penalizes spurious downward vertical velocity while resting on solid ground:

$$\mathcal L_{\text{contact}}(\theta) = \frac{1}{B} \sum_{i=1}^B \mathbb{I}(c_{\text{ground}, t}^{(i)} = 1 \land a_{\text{jump}, t}^{(i)} = 0) \cdot (\hat v_{y,t+1}^{(i)})^2$$

### 6.5 The Optimization Dilemma in Discrete Soft PINNs
Our experiments uncovered a critical theoretical pathology:
- In continuous Partial Differential Equations (PDEs), PINN loss terms operate on continuous gradients derived via *autograd*, yielding smooth loss surfaces.
- In **stiff discrete dynamical systems at 60 Hz**, kinematic residuals reach magnitudes of $\sim 10^5$, vastly outstripping data error ($\sim 10^1$).
- This causes severe **gradient stiffness**: AdamW allocates almost the entire gradient norm to reconciling kinematic integration, hindering convergence on force residuals and contact flags. This explains why **Soft PINNs often perform worse than unregularized MLPs in discrete gaming systems**.

---
