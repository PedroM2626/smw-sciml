[← 10.17](10-17-synchronized-multi-model-visualization-real-snes-vs-pinn.md) · [Contents](../../README.md#contents) · [10.19 →](10-19-autonomous-multi-entity-mpc-planning-on-real-snes.md)

### 10.18 Genuine Multi-Entity Dataset & Supervised Hazard Dynamics Training

To eliminate synthetic assumptions regarding enemy behavior, we engineered an authentic data recording protocol (`scripts/record_multi_entity_gameplay.py`) that executes 35 continuous gameplay episodes against live Rex entities in *Yoshi's Island 1*:
- **Dataset Scale:** **19,702 genuine 12D transitions** saved to `data/raw/smw_multi_entity_dataset.npz` (recorded at 2,706.8 FPS).
- **Rex Interaction Telemetry:** 6,313 transitions (32.0%) captured active proximity and interaction with Rex sprites (`$7E:14C8`, status $\ge 8$).
- **Supervised Model Optimization:** `src/training/train_multi_entity.py` trains `hazard_net` in `MultiEntityPINNDynamics` using AdamW and ReduceLROnPlateau over an 80/20 train/test split.
- **Test Loss:** Converged to a test MSE of **80.3929** for relative displacement residuals, while maintaining **0.0% analytical kinematic violation** on relative entity motion ($\|\Delta\hat X_{t+1} - (\Delta X_t + (\hat v_{xh, t+1} - \hat v_{xm, t+1})/16.0)\|^2 = 0.0$).
- **Weights Preserved:** `results/checkpoints/pinn_multi_entity_best.pt`.

---
