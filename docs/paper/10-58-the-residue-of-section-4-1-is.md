[← 10.57](10-57-what-section-4-costs-the-corrected-physics.md) · [Contents](../../README.md#contents) · [10.59 →](10-59-the-boundary-channel-10-58-predicted-built.md)

### 10.58 The Residue of Section 4.1 Is a Clamp, Not a Noise Term

Sections 10.53 and 10.57 quote a held-out position error of 0.1056 px that is identical across four unrelated models, and read it as the recording's own residue - something no weights are responsible for and nothing can remove. This section measures what that property is, because the answer decides a model-class question the repository has been asked about: whether a stochastic differential term (drift plus diffusion), a jump process, or a PDE-flavoured continuum is the right next object. It measures the residue of §4.1 on 55,085 transitions from 6 committed recordings, in the WRAM units the engine computes in.

| recording | transitions | distinct residue values | off-integer deviation (sub-px) | exactly zero | median exception (sub-px) | mean run (frames) | longest run |
|:--|--:|--:|--:|:--:|--:|--:|--:|
| `gameplay` | 8,077 | 13 | 0.000000 | 92.93% | 29.0 | 13.93 | 44 |
| `jump` | 6,116 | 6 | 0.000000 | 95.08% | 36.0 | 33.44 | 44 |
| `sprint` | 7,253 | 5 | 0.000000 | 93.41% | 36.0 | 22.76 | 43 |
| `multi_entity` | 19,702 | 11 | 0.000000 | 98.10% | 21.0 | 4.16 | 43 |
| `tilemap` | 10,357 | 10 | 0.000000 | 92.71% | 35.0 | 15.73 | 43 |
| `set_multi_entity` | 3,580 | 6 | 0.000000 | 95.78% | 22.0 | 5.59 | 43 |

| recording | exceptions | pinned share | residue = -v on pinned | exception rate pinned | exception rate not pinned | left after clamp |
|:--|--:|--:|--:|--:|--:|--:|
| `gameplay` | 571 | 94.92% | 1.0000 | 1.0000 | 0.0039 | 29 |
| `jump` | 301 | 99.34% | 1.0000 | 1.0000 | 0.0003 | 2 |
| `sprint` | 478 | 96.86% | 1.0000 | 1.0000 | 0.0022 | 15 |
| `multi_entity` | 374 | 75.40% | 1.0000 | 1.0000 | 0.0046 | 89 |
| `tilemap` | 755 | 95.23% | 1.0000 | 1.0000 | 0.0037 | 36 |
| `set_multi_entity` | 151 | 82.78% | 1.0000 | 1.0000 | 0.0075 | 26 |

| recording | horizon (frames) | starts | Gaussian | iid jump | Markov (geometric runs) | renewal (empirical runs) | observed sd (sub-px) |
|:--|:--|--:|:--|:--|:--|:--|:--|
| `gameplay` | 1 | 1,500 | 0.925 (23.9) | 0.954 (20.7) | 0.995 (2.7) | 0.990 (1.6) | 7.2 |
| `gameplay` | 15 | 1,500 | 0.897 (92.4) | 0.897 (81.7) | 0.820 (34.2) | 0.959 (338.8) | 114.7 |
| `gameplay` | 60 | 1,500 | 0.061 (184.7) | 0.057 (181.7) | 0.587 (66.2) | 0.970 (1232.9) | 317.4 |
| `jump` | 1 | 1,500 | 0.957 (25.3) | 0.960 (13.6) | 0.996 (0.1) | 0.996 (0.1) | 7.2 |
| `jump` | 15 | 1,500 | 0.938 (98.1) | 0.938 (72.9) | 0.929 (4.6) | 0.981 (59.3) | 112.5 |
| `jump` | 60 | 1,500 | 0.031 (196.1) | 0.298 (193.1) | 0.789 (44.2) | 0.919 (145.3) | 352.6 |
| `sprint` | 1 | 1,500 | 0.940 (28.8) | 0.962 (34.8) | 0.995 (1.4) | 0.994 (0.1) | 8.5 |
| `sprint` | 15 | 1,500 | 0.919 (111.7) | 0.919 (106.8) | 0.879 (11.5) | 0.983 (429.0) | 126.5 |
| `sprint` | 60 | 1,500 | 0.051 (223.3) | 0.040 (217.0) | 0.693 (52.1) | 0.975 (1485.1) | 362.2 |
| `multi_entity` | 1 | 1,500 | 0.979 (10.4) | 0.979 (0.0) | 0.995 (0.9) | 0.995 (0.9) | 3.1 |
| `multi_entity` | 15 | 1,500 | 0.960 (40.2) | 0.966 (32.1) | 0.941 (20.7) | 0.987 (187.3) | 37.3 |
| `multi_entity` | 60 | 1,500 | 0.917 (80.3) | 0.931 (74.8) | 0.788 (48.0) | 0.983 (783.4) | 109.0 |
| `tilemap` | 1 | 1,500 | 0.935 (24.5) | 0.948 (20.0) | 0.997 (2.6) | 0.994 (2.3) | 7.9 |
| `tilemap` | 15 | 1,500 | 0.905 (94.7) | 0.904 (81.3) | 0.853 (32.0) | 0.951 (330.0) | 109.4 |
| `tilemap` | 60 | 1,500 | 0.051 (189.4) | 0.056 (187.3) | 0.587 (71.8) | 0.923 (1192.2) | 343.0 |
| `set_multi_entity` | 1 | 1,500 | 0.964 (16.8) | 0.964 (0.1) | 0.993 (2.0) | 0.993 (2.0) | 4.7 |
| `set_multi_entity` | 15 | 1,500 | 0.923 (65.2) | 0.929 (55.1) | 0.855 (33.0) | 0.971 (238.3) | 71.0 |
| `set_multi_entity` | 60 | 1,500 | 0.860 (130.3) | 0.863 (128.3) | 0.589 (75.4) | 0.946 (976.3) | 216.6 |

| recording | ranker | features | AUC | base rate |
|:--|:--|--:|:--:|:--:|
| `gameplay` | state + action | 11 | 0.8001 | 7.07% |
| `gameplay` | state + action + previous residue | 12 | 0.9701 | 7.07% |
| `jump` | state + action | 11 | 0.6409 | 4.92% |
| `jump` | state + action + previous residue | 12 | 0.9921 | 4.92% |
| `sprint` | state + action | 11 | 0.6425 | 6.59% |
| `sprint` | state + action + previous residue | 12 | 0.9852 | 6.59% |
| `multi_entity` | state + action | 11 | 0.6360 | 1.90% |
| `multi_entity` | state + action + previous residue | 12 | 0.8372 | 1.90% |
| `tilemap` | state + action | 11 | 0.8566 | 7.29% |
| `tilemap` | state + action + previous residue | 12 | 0.9718 | 7.29% |
| `tilemap` | state + action + tile_patches | 60 | 0.9265 | 7.29% |
| `tilemap` | state + action + next_tile_patches | 60 | 0.9272 | 7.29% |
| `set_multi_entity` | state + action | 11 | 0.9111 | 4.22% |
| `set_multi_entity` | state + action + previous residue | 12 | 0.9541 | 4.22% |
| `set_multi_entity` | state + action + entities | 20 | 0.9270 | 4.22% |
| `set_multi_entity` | state + action + next_entities | 20 | 0.9273 | 4.22% |

1. **The residue is a lattice, so a density is the wrong object.** It takes 5 to 13 distinct values per recording - one of them zero, holding 92.71% to 98.10% of the frames - and the largest departure from an exact integer number of sub-pixels across all 55,085 transitions is 0.000000. A diffusion term would spread probability over the gaps between atoms, and the console never lands there.
2. **What is left is a state constraint: a clamp and a one-pixel reposition.** 75.40% to 99.34% of the exception frames are ones where the recorded position does not move at all while the velocity byte holds a nonzero value, and on every one of those frames in every recording the residue is exactly minus that velocity. Of the 2627 exception frames that sit inside an episode the clamp accounts for 2430; of the 197 it leaves over, every single one is off the velocity by exactly one whole pixel - not a rounding: the accumulator byte this repository reads (10.3's `$7E:13DA$`) counts 256 units to the pixel and is already inside the recorded position, so no carry of itself can move a body a pixel in one frame. Off the clamp the exception rate is 0.034% to 0.755%, against 1.90% to 7.29% in the two-state view, so these recordings contain no frame whose departure is left unexplained. On the gameplay recording the mean absolute residue is 0.1201 px where §10.53 quoted 0.1056 px: the same object measured on a different subset and through a model's predictions, agreeing in size and now explained.
3. **The residue has a memory, which is what a clamp looks like from outside.** The rate of an exception after another exception is 170x to 626x the rate after a clean frame, with mean runs of 4.16 to 33.44 frames. Fitted as iid noise with the per-frame variance the recordings show, the spread after 60 frames is under-predicted by 4.46x to 5.95x (the range over every horizon is 0.92x to 5.95x, so the iid fit is reasonable only at one frame), so an SDE whose volatility is estimated from the residue mis-states the tail by a factor of six at the horizon a planner actually uses.
4. **The four propagators rank as the mechanism predicts.** Mean 90% coverage over the horizons: drift plus iid Gaussian diffusion 0.811, iid empirical jumps 0.824, a two-state chain with geometric run lengths 0.861, and a renewal process that samples run lengths from the recording itself 0.972, against the 0.90 target. On the Winkler interval score the renewal process wins 4 of 6 recordings and the iid jump wins the one whose exceptions are shortest (multi_entity: jump_iid, set_multi_entity: jump_iid). The ordering is the finding: the only stochastic description that works is one that models the *duration* of a clamp, which is a deterministic fact about geometry, not a random draw.
5. **The observable world nearly predicts it, but not from the state the models are given.** A linear ranker on the 8D state plus the action reaches AUC 0.636 to 0.927; adding the previous frame's residue reaches 0.837 to 0.992. The tilemap recording is the one place extra observables move the number on their own - 0.857 to 0.927 - because the tile patches are read through the camera, which is the variable that is missing from the state: the clamp happens at a boundary, and no channel of the eight says where the boundary is.

**What this changes, and what it does not.** It does not make a diffusion or a PDE the right next model class - the lattice, the memory and the exact ±1 px repositions argue against both, and 10.55 already measured the engine as first-order once contacts are excluded. What it does say is that the repository has been paying an irreducible-looking position error for a constraint it never represented: every hard shell and every projection here clamps *velocity* (±72, ±64, and 10.51's bounds projected onto the output), and nothing clamps *position*, which is the channel the engine actually stops. The cheap next step is therefore not a stochastic formulation but a ninth state channel - the boundary the position is pinned against, or the camera offset that implies it - and the prediction this section supplies for it is sharp: adding it should remove the residue that §10.53 called the console's own, and leave the whole-pixel repositions as the only exception left. **10.59 built that channel, tested the prediction, and it failed**: the camera address was identified from the console and recorded next to the state, the screen-edge channel covers 1.00% of the exception frames, and the frames this section read as a clamp turn out to be whole player records repeating on a console that is still running - a paused simulation, not a stopped body. The numbers in this section are what the recordings measure and stand; the word *clamp* is what 10.59 withdraws.

**Limitations.** The clamp is inferred from the recorded position not moving while the velocity byte holds, not read from the engine's collision response, so its mechanism (level edge, camera lock, pipe or warp) is attributed rather than observed - the recordings carry no camera or level-bound channel, which is the same observation gap this section names. 10.59 closed that gap and the attribution did not survive it: with the camera, the mode byte and a WRAM CRC recorded, the mechanism reads as the engine not processing the player object rather than as a boundary acting on it. The vertical identity is not scored here: §4.3.5's clamp is its own mechanism, measured by 10.52/10.54/10.56, and the horizontal residue is what the 0.1056 px figure came from. The residue is computed on the carried velocity, the convention §10.53 measured and §10.54 records, and §10.57's flags make the other reading available without changing these numbers. Coverage is scored on recorded trajectories under the actions the player actually took, so it is a statement about the transition kernel and not about any policy's state visitation; the renewal kernel samples run lengths independently of which boundary caused the run, which is why it over-covers at short horizons. The rankers are linear ridge fits, so they bound what a linear model sees, not what is learnable.

Regenerate: `python -m src.evaluation.residue_process_study` (emulator-free, reads the six committed recordings, ~3 min; `make residue-process`; smoke config `configs/smoke_residue_process.yaml`; `--recordings`, `--samples` and `--horizons` control the budget). Writes `results/residue_process_metrics.json`.
