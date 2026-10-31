[← 10.38](10-38-closed-loop-reproduction-audit-and-preamble-probe.md) · [Contents](../../README.md#contents) · [10.40 →](10-40-physics-parameter-identification-the-inverse-problem.md)

### 10.39 Physics-Informed Model-Free RL (PIML-MFRL)

Every closed-loop controller up to now split into two camps: model-based planners that
roll the learned world model forward (Sections 10.6-10.14) and *model-free* PPO that
learns only from real console transitions (Section 10.9). The natural third position -
keep the agent model-free but let the known engine physics shape its **critic**, its
**action** or its **objective** - was unbuilt. PIML-MFRL (`src/training/piml_mfrl.py`)
trains standard PPO directly on the authentic SNES console (`SnesSingleEnv`) and couples
the Section 4 physics through three independently switchable mechanisms, none of which
predicts $s_{t+1}$, so the critic never stops being a genuine model-free value
estimator.

**Approach A - Physics-Informed Critic** (`--use-physics-critic`). The optimal value of
a physical system is not an arbitrary statistic: as the agent falls toward a pit it must
decay like a Control-Lyapunov function. `PhysicsInformedCriticLoss`
(`src/losses/physics_rl_losses.py`) adds

$$\mathcal L_{\text{critic}} = \mathcal L_{\text{TD}} + \lambda_c \, \mathbb E_{s\in\mathcal D_{\text{danger}}}\big[\, \mathrm{relu}\big(\nabla_s V_\phi(s)\cdot \dot{s} + \alpha\, V_\phi(s)\big)\, \big]$$

where $\dot{s}$ is the *analytic* discrete drift of Section 4 ($\dot{x}=v_x/16$,
$\dot{y}=v_y/16$, gravity on $\dot v_y$), never a rolled-out prediction, and
$\mathcal D_{\text{danger}}$ is the falling-toward-pit slice. $\nabla_s V$ is taken with
a differentiable backward pass (`create_graph=True`).

**Approach B - Physics-Constrained Actor (CBF filter)** (`--use-cbf-filter`). The actor
proposes action logits; a differentiable safety layer projects the *policy* onto the
feasible action set before it reaches the console. `src/models/cbf_projection.py`
ships both halves: `CBFQPLayer` solves the continuous projection
$\min_a \lVert a-\hat{a}\rVert^2$ s.t. $b_i(s)\,a \ge c_i(s)$ (exact in closed form for
one active constraint, a convergent POCS sweep for several) with the velocity-saturation
barrier `smw_barrier_affine`; because SNES exposes 8 macro-actions, the trainer uses its
categorical analogue `DiscreteCBFCategoricalFilter`, `logits_safe = logits - beta * R_phys`,
whose $\beta\to\infty$ limit is a hard filter that never samples an infeasible action.

**Approach C - Physical regularisation of the PPO surrogate** (`--use-action-penalty`).
`ActionPhysicsViolation` charges the surrogate loss the physical residual, in expectation over the
policy's own state-action distribution:

$$\mathcal L_{\text{PPO}} \mathrel{+}= \lambda_a\,\mathbb E_{(s,a)\sim\pi_\theta}[\,\mathcal R_{\text{phys}}(s,a)\,]$$

where $\mathcal R_{\text{phys}}$
counts the *force the engine resolves to zero*: thrust into a wall already flagged by
`c_left`/`c_right`, a rise into a flagged ceiling, or acceleration demanding $|v_x|$ beyond
the hardware ceiling. This is the same safety model the CBF filter re-weights by.

| Mechanism | Acts on | Stays model-free because | Code |
| :--- | :--- | :--- | :--- |
| A - Critic Lyapunov/HJB | evaluation | scores $V(s)$ on real states, $\dot{s}$ is analytic kinematics, no $s_{t+1}$ roll-out | `PhysicsInformedCriticLoss` |
| B - CBF actor filter | execution | projects the policy, does not simulate the transition | `CBFQPLayer`, `DiscreteCBFCategoricalFilter` |
| C - Surrogate penalty | structure | penalises the commanded action's impossible force demand | `ActionPhysicsViolation` |

**Cost.** The mechanisms add only a few tensor operations per mini-batch (A also one
extra critic forward and a second-order grad), so wall-clock overhead against the Section
10.9 model-free baseline is small; the physics does not require training a world model or
running the emulator any differently.

**Per-mechanism correctness** (Lyapunov decay, CBF projection identity and feasibility,
violation-penalty sign, and the full A+B+C update path) is covered by emulator-free unit +
integration tests in `tests/test_piml_mfrl.py`, which run in CI against a mock console.

#### 10.39.1 Measured Study on Real Hardware

`src/evaluation/piml_mfrl_study.py` (`smw-pinn piml-mfrl-study`) is a **per-mechanism
ablation**: it runs the same model-free PPO loop on Yoshi's Island 1 over 3 seeds
(42/43/44) at a matched 10,000-console-frame budget for every coupling in isolation and
in combination, so no mechanism is left out and any effect is attributed to a specific
coupling rather than a black-box on/off toggle (`results/piml_mfrl_metrics.json`, RTX 4070
Laptop GPU):

| Condition (switches) | Mean final return (± std, n=3) | Δ vs baseline | Executed-action violation | Mean training time |
| :--- | :---: | :---: | :---: | :---: |
| **Model-free PPO (baseline, A/B/C off)** | **1330.0 ± 267.4** | — | 0.0000 | 21.9 s |
| A only (physics-informed critic) | 1317.1 ± 284.1 | −1.0% | 0.0000 | 25.7 s |
| B only (CBF-QP actor filter) | 1342.7 ± 287.0 | +1.0% | 0.0000 | 29.2 s |
| C only (surrogate action penalty) | 1330.0 ± 267.4 | 0.0% (identical) | 0.0004 | 24.3 s |
| A+B+C (full PIML-MFRL) | 1300.8 ± 292.3 | −2.2% | 0.0001 | 33.4 s |

**Honest reading - a null result, localised by the ablation.** Every return delta sits
far inside the ± ~270-290 seed standard deviation (largest, the full stack, is −2.2%), so
no coupling is distinguishable from the canonical model-free baseline at this budget on
this stage. The per-mechanism view sharpens *why*: the executed-action physics violation is
~0 for **every** condition (0.0000-0.0004), so the CBF filter (B) and the surrogate penalty
(C) have nothing to correct on open ground and are correctly *inert*. Approach C is the
cleanest demonstration of this - it reproduces the baseline return **exactly on all three
seeds** (1368.25 / 985.05 / 1636.65), because its only gradient contribution is
$\lambda_C \cdot \text{violation}$ and that term is ~0, leaving the policy update bit-for-bit
unchanged. Approach B is the sole condition to move the needle at all, and only inside
noise (best single seed 1688.3, the study maximum, at $\Delta = +1.0\%$). The one unambiguous
axis is **cost**: training time rises monotonically with the number of active couplings
(21.9 s baseline to 33.4 s full, ≈1.5×), the critic Lyapunov term adding a second
critic pass and a second-order gradient per mini-batch.

The ablation therefore *localises* where physics-coupled model-free RL should pay off -
environments or curricula where infeasible actions are common (hazard stages, narrow pipe
geometry, dense sprite threats) - instead of over-claiming a gain this stage cannot
exhibit, and shows the null is a property of the task's geometry, not of a broken coupling.
This is consistent with Section 10.30: the benefit of a structural prior scales with how
often the unconstrained learner is tempted to violate it.

![PIML-MFRL per-mechanism ablation](../../results/figures/piml_mfrl_comparison.png)

Regenerate: `python -m src.evaluation.piml_mfrl_study --seeds 42,43,44 --total-timesteps 10000`.
