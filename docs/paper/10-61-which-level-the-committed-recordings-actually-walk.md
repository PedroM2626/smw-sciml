[← 10.60](10-60-the-dashboard-at-the-top-of-this.md) · [Contents](../../README.md#contents) · [10.62 →](10-62-the-pixel-recording-re-made-from-a-level.md)

### 10.61 Which Level the Committed Recordings Actually Walk

10.36.3 closed with a question it could not answer: the terrain reconstructed from the recorded 7x7 tile patches did not match the terrain of the genuinely entered level, and that left two explanations - the recordings are of a different stage, or the tile buffer of the forced-mode session is not a level buffer at all. The figure quoted there came from a scratch run with an undefined sampling rule and no artifact, so it is superseded here rather than re-derived. The profile is one instrument applied three times: first non-air tile at or below Mario's own tile row, within the 7x7 window the recording carries, modal value per absolute tile column.

| side | observations | columns seen | x range (px) | message boxes cleared |
| --- | --- | --- | --- | --- |
| recording `data/raw/smw_tilemap_dataset.npz` | 10,357 | 61 | 6.81 -> 918.94 | - |
| forced-mode session, live | 895 | 71 | 16.00 -> 1079.69 | 2 |
| verified level entry, live | 900 | 87 | 16.00 -> 1334.75 | 2 |

Scored pairwise, on the columns both sides see and within 1 row:

| comparison | columns | agreeing | share |
| --- | --- | --- | --- |
| recording vs genuine level | 61 | 38 | 62.30% |
| recording vs its own session, live | 61 | 58 | 95.08% |
| forced-mode session vs genuine level | 71 | 38 | 53.52% |

The two live arms run the identical scripted walk - hold right and dash, jump every 24 frames, pulse A at a held player - so their comparison is the one that isolates terrain: they agree on 53.52% of the 71 columns both cover. The forced-mode session is therefore walking a buffer that is not the terrain of the stage the world-map route enters, and the second row says the instrument is not what separates them: replaying it live over the session the file came from reproduces the committed recording on 95.08% of columns. What the recordings describe is the forced-mode session, faithfully. What the forced-mode session *is* stays open at this resolution - another stage's buffer, or one left stale by the boot - and 10.36.3 already settled its visual half, which is that the PPU never leaves the menu.

**Limitations.** The profile is measured from Mario's own tile row, so it answers "what is under him" rather than "what does the level look like", and the window is only 7 tiles deep: a column is observed only while its ground is inside that window, which is why the three sides see different column counts. Two walks over the same terrain can still disagree where one Mario stands on a block and the other on the ground, so every share here is a floor on the similarity, not an estimate of it - and the forced-mode arm ended its walk in mode 0x17 while the level arm was still in 0x14, which is geometry, not sampling.
