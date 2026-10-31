[← 10.31](10-31-end-to-end-pixel-perception-pixel-to-action-front-end.md) · [Contents](../../README.md#contents) · [10.33 →](10-33-mpc-reflex-ablation-pure-vs-reflexive-td-mpc.md)

### 10.32 Hierarchical Global + Local Planning (A* + CEM-MPC)

To close limitation §10.30-5 (local MPC minima at vertical obstacles),
`src/planning/global_planner.py` builds the global occupancy grid from the
WRAM tile buffer and runs 8-connected A* (no corner-cutting, climb penalty
approximating jump effort, hazard costs) to extract pixel waypoints, tracked
by a local 15-frame Hard-PINN CEM-MPC via `WaypointObjective`
(`HierarchicalMPCController`, `--value-ckpt` flag for TD-MPC mode in
`src/evaluation/evaluate_hierarchical_mpc.py`). Division of labor is explicit:
A* gives topological guidance, the local MPC owns jump-arc feasibility.

#### 10.32.1 Measured Result on Real Hardware

`results/hierarchical_mpc_metrics.json`: A* routed **511** tiles into **8** pixel
waypoints; the local MPC reached **2 of 8** waypoints and **107.88 px** before
terminating at frame **201**. The route is not the bottleneck - the local planner
still fails to produce a feasible jump arc at the second waypoint, so global
guidance alone does not close the gap that §10.33 quantifies with hand-coded
reflexes (1,065 px with reflexes vs 114 px without). This row is a negative result
and is reported as such.
