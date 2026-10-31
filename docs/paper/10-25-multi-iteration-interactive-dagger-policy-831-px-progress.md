[← 10.24](10-24-autonomous-extended-level-navigation-on-real-snes.md) · [Contents](../../README.md#contents) · [10.26 →](10-26-comprehensive-ablation-study-clamping-horizon-drift-cem.md)

### 10.25 Multi-Iteration Interactive DAgger Policy (831 px Progress & 2,707 FPS)

While single-step Behavioral Cloning achieved 115 px before suffering from compounding drift, deploying the full **DAgger** algorithm (*Dataset Aggregation*, Ross & Bagnell, 2011) over 3 interactive on-policy iterations (`src/training/train_dagger.py`) completely closed the imitation gap:
- **Iteration 1 (BC seed):** 115.0 px progress (pit fall at frame 174).
- **Iteration 2 (On-policy corrective queries):** Jumped to **889.1 px** progress!
- **Iteration 3 (Fine-tuning on boundary states):** Stabilized at **831.8 px** progress.
- **Hardware Validation (`results/dagger_policy_metrics.json`):**
  - **Survival:** **500 / 500 frames** (100% survival rate).
  - **Hardware Progress:** **831.75 pixels** on live console emulation.
  - **Inference Latency:** **369.34 µs / step** on CPU.
  - **Inference Throughput:** **2,707.5 FPS** (a **114× acceleration** over online CEM MPC at 23.7 FPS).
- **Artifacts:** Checkpoint saved to `results/checkpoints/dagger_policy_best.pt`.

---
