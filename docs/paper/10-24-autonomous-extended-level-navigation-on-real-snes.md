[← 10.23](10-23-amortized-policy-distillation-from-live-mpc-decisions.md) · [Contents](../../README.md#contents) · [10.25 →](10-25-multi-iteration-interactive-dagger-policy-831-px-progress.md)

### 10.24 Autonomous Extended Level Navigation on Real SNES Hardware (1,016+ px Progress)

To test the multi-entity world model beyond localized obstacle evasion, we deployed an extended horizon benchmark (`src/evaluation/evaluate_extended_navigation.py`) spanning up to 1,500 frames on live SNES hardware:
- **Milestones Cleared:**
  - Milestone 500 px cleared at frame 264.
  - Milestone 782 px cleared at frame 407.
  - **Milestone 1,000 px cleared at frame 512!**
- **Final Metrics:**
  - **Total Progress:** **1,016.06 pixels** (a **29.8% increase** over the previous 782 px record).
  - **Survived Frames:** **627 frames** (over 10.4 seconds of continuous authentic 60 Hz gameplay).
  - **Multi-Obstacle Transposition:** Successfully evaded Rex 1, scaled the diagonal solid hill at $X \approx 828$, and leaped past Rex 2 into the 1,000+ px territory.
  - **Throughput:** Maintained 25.26 FPS real-time planning throughput on GPU.
- **Visual Evidence:** Multi-panel publication figure saved to `results/figures/extended_level_navigation.png` and raw telemetry to `results/extended_navigation_metrics.json`.

![Extended Level Navigation](../../results/figures/extended_level_navigation.png)

---
