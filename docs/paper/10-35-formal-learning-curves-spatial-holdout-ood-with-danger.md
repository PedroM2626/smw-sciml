[← 10.34](10-34-connected-orphans-tilemap-closed-loop-unified-joint-training.md) · [Contents](../../README.md#contents) · [10.36 →](10-36-yoshi-s-island-2-capture-blocked-with.md)

### 10.35 Formal Learning Curves & Spatial-Holdout OOD with Danger

`src/evaluation/plot_learning_curves.py` builds the Model-Free vs Dyna figure
from committed artifacts only (Dyna has no logged per-step curve, so it
appears as an annotated operating band, never a fabricated curve):
Model-Free PPO needed **39,936 real frames / 103 episodes** to converge
(return 1,652.2); Dyna-PINN operates on **200–8,077 frames**.

Yoshi's Island 2 capture is **blocked** (see §10.36), so OOD-with-danger is
delivered as a spatial holdout on Yoshi's Island 1
(`src/evaluation/spatial_holdout_benchmark.py`, pure `.npz`, runs in CI):
committed checkpoints evaluated zero-shot on X > 700 (2,105 transitions;
far 12D slice has **1,789 live-hazard frames, 78% danger density**):

| Model | In-distribution MSE | Far-region MSE | Far violations | Far drift |
| :--- | :---: | :---: | :---: | :---: |
| Statistical MLP | 16.47 | **43,408.26** (2,600x collapse) | 100.0% | 667.5 px |
| Hard Residual PINN | 0.58 | **27.82** (48x degradation, 1,560x better than MLP) | **0.0%** | 170.2 px |

Honest reading: kinematics transfer (0.0% violations preserved); contact/force
residuals transfer only partially — the next modeling frontier.
