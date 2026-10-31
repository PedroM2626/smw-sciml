[← 10.9](10-09-canonical-model-free-ppo-baseline-vs-pinn-mbrl-sample.md) · [Contents](../../README.md#contents) · [10.11 →](10-11-spatial-translation-invariant-pinn-dynamics.md)

### 10.10 Deep Ensemble of Hard PINNs & Epistemic Uncertainty Quantification

To eliminate *Model Exploitation* (where policies optimize into regions where single neural networks hallucinate optimistic transitions), we engineered a **Deep Ensemble of $E = 5$ Hard Residual PINNs** (`src/models/pinn_ensemble.py`):

$$\mathcal{E} = \{f_{\theta_1}, f_{\theta_2}, \dots, f_{\theta_5}\}$$
trained with independent seed initializations ($\text{seed} \in \{42, 59, 76, 93, 110\}$) across bootstrap partitions of WRAM telemetry.

#### Mathematical Formulation:
- **Predictive Mean Transition:**

  $$\mu(\hat s_{t+1}) = \frac{1}{E} \sum_{e=1}^E f_{\theta_e}(s_t, a_t)$$
- **Epistemic Disagreement Variance:**

  $$\sigma^2(\hat s_{t+1}) = \frac{1}{E-1} \sum_{e=1}^E \|f_{\theta_e}(s_t, a_t) - \mu(\hat s_{t+1})\|^2$$
- **Pessimistic Reward Function (Safe MBRL):**

  $$r_{\text{safe}}(s, a) = r(s, a) - \beta \cdot \sqrt{\sum \sigma^2(\hat s_{t+1})}$$

#### Empirical Results on Out-of-Distribution (OOD) Dynamics Detection:
When evaluated on genuine in-distribution test transitions, the 5 PINN members disagree by $\mu_{\sigma} = 0.4609$. Under non-physical kinematic shocks (extreme velocity perturbations outside the training support) the disagreement does not grow - it measures 0.4189, a ratio of 0.909. As recorded, this ensemble **does not** flag those shocks as out-of-distribution, so the variance trigger this section was written to justify is not supported by its own artifact; Section 12 records the retraction, and the `sigma > tau` rule of 10.15 has to be read as a mechanism that is implemented and tested on synthetic perturbations, not as a detector demonstrated on this data:

| Ensemble Metric | In-Distribution Test Data | Out-of-Distribution Shock Data | OOD Diagnostic Sensitivity |
| :--- | :---: | :---: | :---: |
| **Epistemic Uncertainty ($\sigma$)** | **0.4609** | **0.4189** | Ratio 0.909 - the shock does not raise the spread |
| **Ensemble Training Latency** | 17.3 seconds (all 5 models) | N/A | Highly parallel GPU scaling |

![Ensemble Uncertainty Distribution](../../results/figures/pinn_ensemble_uncertainty.png)

---
