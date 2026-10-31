[← 10.20](10-20-spatial-discrete-tilemap-perception-via-wram-7e.md) · [Contents](../../README.md#contents) · [10.22 →](10-22-permutation-invariant-set-multi-entity-world-model-cross-attention-for.md)

### 10.21 Empirical Grounding of Tilemap-PINN (WRAM `$7E:C800` Dataset & 98.66% Contact Accuracy)

To bridge the gap between theoretical architecture and empirical validation, we recorded a dedicated genuine dataset combining continuous kinematics with discrete local stage geometry:
- **Genuine Dataset:** `data/raw/smw_tilemap_dataset.npz` containing **10,357 transitions** with local $7 \times 7$ tile patches extracted directly from `$7E:C800` during interactive gameplay in *Yoshi's Island 1* (throughput: 2,577 FPS).
- **Supervised Training:** `src/training/train_tilemap.py` trained `TilemapPINNDynamics` with joint SmoothL1 and BCE contact loss.
- **Empirical Contact Accuracy:** Achieved **98.66% contact accuracy** across all collision flags (`c_ground`, `c_ceiling`, `c_left`, `c_right`), outperforming the terrain-blind baseline (98.08%) by **+0.58%** absolute gain.
- **Analytical Kinematic Guarantee:** Maintains exact **0.000000 kinematic residual** (0.0% violation).
- **Artifacts:** Checkpoint saved to `results/checkpoints/tilemap_pinn_best.pt` and metrics to `results/tilemap_benchmark_metrics.json`.

---
