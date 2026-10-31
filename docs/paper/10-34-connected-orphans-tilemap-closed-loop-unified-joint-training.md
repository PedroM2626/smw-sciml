[← 10.33](10-33-mpc-reflex-ablation-pure-vs-reflexive-td-mpc.md) · [Contents](../../README.md#contents) · [10.35 →](10-35-formal-learning-curves-spatial-holdout-ood-with-danger.md)

### 10.34 Connected Orphans: Tilemap Closed-Loop, Unified Joint Training, Set-12

| Orphan | Connection | Measured result |
| :--- | :--- | :---: |
| Tilemap-PINN | `TilemapMPCWrapper` adapts (kinematics, patch, action) to the MPC interface (static-map approximation over horizon H, stated); `src/evaluation/evaluate_tilemap_mpc.py` refreshes the 7x7 patch every frame, **no reflexes** | 400/400 frames, **808.6 px** (vs 114.1 px pure 8D MPC) |
| Unified Multimodal PINN | `src/training/train_unified_multimodal.py` alternates genuine tilemap batches (kinematics + patch + contact BCE) and multi-entity batches (hazard MSE, active-masked) | 5 epochs: train 1.04→0.89, val 1.06→0.95 (`unified_joint_best.pt`) |
| Set-Multi-Entity (K=12) | `src/environment/sprite_sets.py` (shared slot-to-row conversion) + `scripts/record_set_multi_entity_gameplay.py` (all active sprites) + `src/training/train_set_multi_entity.py` (target-active-masked loss) | 3,580 genuine transitions (2,235 with live sprites); 5 epochs val 0.54→0.45 |
| Vertical physics ID | `GravityIdentifiedPINNDynamics`: exact integrator + residual head + **learnable** $g_{\text{hold}}$ / $g_{\text{fall}}$ (init 3.0/6.0); unit test recovers $g_{\text{hold}} = 3$ from synthetic arcs | structural, tested |
