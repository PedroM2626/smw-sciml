[← 10.27](10-27-master-algorithm-comparison-table-world-models-mpc.md) · [Contents](../../README.md#contents) · [10.29 →](10-29-consolidation-of-frontiers-a-b-c-and.md)

### 10.28 Frontier 2: Zero-Shot Closed-Loop Control on Unseen Stage B (*Yoshi's House*)

To validate conclusively whether the physical knowledge embedded in the **Hard Residual PINN** and in the **DAgger** policy generalizes to new environments without suffering from *overfitting* or out-of-distribution (OOD) collapse, we submitted every controller to the closed-loop fire test on the unseen stage **Yoshi's House** (`$7E:0100 = 0x14`, `data/raw/smw_yoshi_house.state`).

No model, planner or network received any training sample, fine-tuning or calibration on Yoshi's House. Mario was initialized at coordinate $X_0 = 16.0, Y_0 = 336.38$, and each controller was re-seeded and run autonomously for 400 frames at 60 Hz over 5 seeds (42-46); every metric below is a mean ± standard deviation across those seeds (`src/evaluation/evaluate_cross_level_control.py`):

#### 10.28.1 Empirical Results Obtained on the Real SNES Console

The results stored in `results/cross_level_control_metrics.json` reveal the robustness of the structured formulation:

| Controller | Horizontal Progress ($X$) | Survival (frames) | Mean Velocity ($\bar v_x$) | Decision Latency (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Actions Baseline** | 183.40 ± 35.71 | 400 ± 0 | 7.35 ± 1.43 | 0.023 | 43,813 ± 2,519 |
| **MPC + Statistical MLP (Black-Box OOD)** | 44.38 ± 22.01 | 342 ± 33 | 2.61 ± 0.62 | 20.61 ± 5.37 | 51.1 ± 9.9 |
| **MPC + Soft-Constrained PINN** | 88.05 ± 23.63 | 400 ± 0 | 3.52 ± 0.94 | 21.95 ± 7.21 | 49.1 ± 11.0 |
| **MPC + Hard Residual PINN (Ours)** | **522.88 ± 335.33** | 309 ± 111 | **26.56 ± 6.68** | 24.03 ± 1.70 | 41.8 ± 3.1 |
| **Amortized DAgger Policy (Ours)** | **830.50 ± 0.00** | **400 ± 0** | **34.66 ± 0.00** | **0.379 ± 0.028** | **2,654 ± 200** |

*All figures are means over 5 seeds (42-46) from the re-recorded multi-seed protocol; latency and throughput are wall-clock and therefore machine dependent. The DAgger row is ±0 because that reactive policy is fully deterministic.*

#### 10.28.2 Comparative Analysis and Scientific Conclusions

1. **Monotonic benefit of the hard kinematic constraint under zero-shot transfer:** With 5 seeds the MPC ordering is clean and monotonic in physical fidelity - Statistical MLP (**44.38 px**) < Soft-Constrained PINN (**88.05 px**) < Hard Residual PINN (**522.88 px**). The hard model reaches 5.9× the soft model and 11.8× the black-box MLP, because enforcing $(\Delta x = v_x \Delta t)$ in the output layer stops the CEM planner from exploiting model error on a stage it never saw.
2. **The Hard PINN clears the unseen obstacle, seed-dependently:** Its per-seed progress is bimodal - 785, 784 and 820 px on three seeds (it surmounts the Yoshi's House geometry and runs like the DAgger policy) versus 111 and 114 px on two seeds (it stalls and dies at ≈173 frames, hence the 309-frame mean survival). Read as an *ordering* (Section 10.38.2) the result is unambiguous: only the hard-constrained planner ever clears the stage, and even its weakest seed matches the soft model's best seed, while no MLP or soft run clears it.
3. **A mis-specified OOD model is worse than acting at random:** The MLP and Soft MPC runs (44 and 88 px) fall *below* the Random baseline (183 px) and the MLP even loses lives (342-frame survival). This is the closed-loop signature of the gradient conflict predicted in Section 4: planning on an unconstrained black-box or a softly penalised model hallucinates infeasible accelerations on unseen terrain, so committing to its plan hurts relative to unmodelled exploration - the hard constraint is what removes the failure mode.
4. **Efficiency and reproducibility of the amortised policy:** The **DAgger** policy leads in distance (**830.50 px**) and is the only controller that reproduces *exactly* across all seeds (±0), running at **≈2,654 FPS** with 0.38 ms decision latency because it replaces online CEM with a single forward pass.
5. **Kinematic validation:** The representative altitude trajectory $Y(t)$ (first seed, figure below) shows gravity parabolas with ground contact at $Y \approx 336$ px and no penetration or teleportation.

![Zero-Shot Closed-Loop Control on Stage B](../../results/figures/cross_level_control_trajectories.png)
*Figure: Closed-loop trajectory curves on the Libretro SNES console in the unseen Yoshi's House stage. Top panel: accumulated horizontal progress. Bottom panel: vertical altitude highlighting parabolic jump cycles and rigid ground contact.*

---
