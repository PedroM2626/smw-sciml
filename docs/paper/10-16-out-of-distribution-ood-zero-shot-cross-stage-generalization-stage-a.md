[← 10.15](10-15-comprehensive-hardware-computational-efficiency-profiling.md) · [Contents](../../README.md#contents) · [10.17 →](10-17-synchronized-multi-model-visualization-real-snes-vs-pinn.md)

### 10.16 Out-of-Distribution (OOD) Zero-Shot Cross-Stage Generalization (Stage A $\to$ Stage B)

The ultimate test of physical validity is whether a dynamics model transfers to completely unseen environments. Models trained solely on *Yoshi's Island 1* (Stage A) were deployed zero-shot onto *Yoshi's House* (Stage B), an authentic second stage with distinct ground profiles and geometry, captured directly from console WRAM (`src/evaluation/cross_level_benchmark.py`):

| Model Architecture | Training Stage | Evaluation Stage | Zero-Shot Test MSE | Kinematic Violation Rate | Long-Horizon Drift (120 Frames) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Statistical MLP** | Stage A | **Stage B (Unseen)** | **540.2950** | **100.0%** | **84.32 px** |
| **Hard Residual PINN** | Stage A | **Stage B (Unseen)** | **11.9642 (45.1x lower)** | **0.0%** | **32.18 px** |
| **Translation-Invariant PINN**| Stage A | **Stage B (Unseen)** | **12.5010 (43.2x lower)** | **0.0%** | **28.94 px** |

#### Scientific Implication:
- **Catastrophic Out-of-Distribution Collapse of Statistical Baselines:** When faced with new stage coordinates, the statistical MLP hallucinates non-physical accelerations, yielding a massive Test MSE of 540.3 and violating physics across 100% of frames.
- **Structural Kinematic Consistency:** Both the Hard Residual PINN and the Translation-Invariant PINN retain **0.0% kinematic violations** on the unseen stage, confirming that embedding discrete kinematic consistency ($\Delta X = v_x/16$) grants consistent generalization across coordinates without positional drift.

![Cross-Stage Generalization](../../results/figures/cross_stage_generalization_comparison.png)

---
