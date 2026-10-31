[← 10.32](10-32-hierarchical-global-local-planning-a-cem-mpc.md) · [Contents](../../README.md#contents) · [10.34 →](10-34-connected-orphans-tilemap-closed-loop-unified-joint-training.md)

### 10.33 MPC Reflex Ablation (Pure vs Reflexive) & TD-MPC Terminal Value

Published full-level runs overlay three hand-coded reflexes (wall vault,
hazard vault, B edge-pulse, extracted verbatim into `apply_reflexes`). The
head-to-head ablation on real hardware (`src/evaluation/mpc_reflex_ablation.py`,
600 frames, same savestate) is decisive:

| Condition | Progress (px) | Survived | Termination | Reflex firings |
| :--- | :---: | :---: | :---: | :---: |
| **Pure CEM-MPC (H=16)** | 114.1 | 181 frames | pit fall | 0 |
| **MPC + reflexes** | **1065.2** | **600 frames** | timeout (alive) | 50 hazard + 22 B-pulse + 0 wall |

The principled replacement (TD-MPC paradigm) is implemented, not just
proposed: `TerminalValueObjective` adds a discounted learned terminal value
$\gamma^H V(s_H)$ to waypoint tracking. $V$ was fit by Monte-Carlo regression
on the genuine 971-frame clearance log (`src/training/train_terminal_value.py`,
in-sample $R^2 = 0.80$, checkpoint `results/checkpoints/terminal_value_best.pt`).
