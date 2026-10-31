[← 10.22](10-22-permutation-invariant-set-multi-entity-world-model-cross-attention-for.md) · [Contents](../../README.md#contents) · [10.24 →](10-24-autonomous-extended-level-navigation-on-real-snes.md)

### 10.23 Amortized Policy Distillation from Live MPC Decisions (2,900 FPS vs 23.7 FPS)

To solve the Sim-to-Real Objective Mismatch failure of Dyna-PPO (which scored $-7.38\text{ px}$ due to model exploitation) without incurring the heavy inference cost of online CEM MPC (23.7 FPS):
- **Expert Hardware Demonstrations:** `src/training/distill_mpc_policy.py` executed closed-loop MPC on authentic SNES emulation to gather 1,600 expert state-action pairs $(s_t, a_t^*)$ across 4 successful episodes (mean progress $\sim 764\text{ px}$).
- **Supervised Policy Distillation:** Trained a compact feedforward `DistilledActorPolicy` via behavioral cloning with BCE loss (converging to 0.1740).
- **Inference Throughput Benchmark (`src/evaluation/evaluate_distilled_policy_snes.py`):**
  - **Inference Latency:** **344.73 µs / step** on CPU.
  - **Policy Throughput:** **2,900.9 FPS** (a **122× acceleration** over online CEM MPC at 23.7 FPS).
  - **Console Validation:** Surpassed the prior Dyna-PPO model collapse, advancing forward to **115.0 px** and surviving 174 frames without retrograde motion.
  - **Artifacts:** Checkpoint saved to `results/checkpoints/distilled_mpc_policy.pt` and metrics to `results/distilled_policy_metrics.json`.

---
