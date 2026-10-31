[← 10.63](10-63-the-held-player-has-two-mechanisms-and.md) · [Contents](../../README.md#contents) · [10.65 →](10-65-velocity-from-frames-that-actually-move.md)

### 10.64 The Buffer the Telemetry Reads Is the Title Screen's Own

10.61 could say the recordings' terrain is not the entered level's, and could not say what it is. The boot answers that: at frame 410 of the published preamble the game is sitting on its title screen with the attract demo running - mode `0x07`, translevel `0x00`, the demo's own player at x moving under a camera that follows it - and the preamble writes `0x14` into that session without loading anything. So the buffer every recording reads is the title screen's.

| session | mode at start | mode at end | translevel | x travelled (px) | columns sampled |
| --- | --- | --- | --- | --- | --- |
| attract demo, no input | `0x07` | `0x07` | `0x00` | 1254.94 | 79 |
| forced-mode preamble | `0x14` | `0x17` | `0x00` | 1016.69 | 64 |
| entered level (10.36.3) | `0x14` | `0x14` | `0x2A` | 1318.75 | 83 |

Scored on the columns each pair visited, cell by cell over all 27 rows of a column, and again by 10.61's class profile:

| pair | columns | identical columns | equal cells | class profile agrees |
| --- | --- | --- | --- | --- |
| forced mode preamble vs attract demo | 64 | 96.88% | 99.88% | 92.54% |
| forced mode preamble vs entered level | 64 | 0.00% | 84.20% | 55.88% |
| attract demo vs entered level | 79 | 1.27% | 84.81% | 54.32% |

Cell equality on its own would mislead, and the middle column is what stops it: most of a column is empty rows, so two unrelated buffers still agree on 84.20% of their cells. Read a column whole instead, and the title session and the forced-mode session are the same buffer - 96.88% of the 64 columns they share are identical row for row - while the entered level shares 0.00% of them. 10.61's class profile says the same thing from the other side: 92.54% against 55.88%. The translevel byte agrees: `0x00` for the demo and for the forced-mode session, `0x2A` for the level the map route enters.

**What this changes, and what it does not.** It does not move a single physics number: the sub-pixel scale, the gravity steps and the velocity bounds the whole of sections 10.37 onward reports were measured out of WRAM telemetry, and the engine was running real physics while it did it. What it settles is what the tile-conditioned results are conditioned *on* - the terrain of the level the demo plays, not of a level the harness chose - and it retires the alternative 10.61 left open, that a different stage's buffer had been loaded. Nobody chose a stage: the title screen did. The entered level remains the only session here whose stage the repository picked, which is why 10.60's dashboard and 10.62's re-recording both run on it.
