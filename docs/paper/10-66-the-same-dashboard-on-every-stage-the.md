[← 10.65](10-65-velocity-from-frames-that-actually-move.md) · [Contents](../../README.md#contents) · [10.67 →](10-67-re-running-the-map-survey-and-the-map.md)

### 10.66 The Same Dashboard on Every Stage the Map Will Actually Give, and What Clears Mean

The hero is one stage, and one stage is what the repository had: 10.36.3 built a savestate by walking the map to the first node past Yoshi's House, and every picture-based claim since has been resting on it. Asked for more levels, the honest answer starts by asking the engine. `scripts/survey_world_map.py` walks the map cursor from that node in all four directions, 7 distinct positions deep, and presses A on each.

| map cursor | reached by | translevel | timer | right-press dx | the engine gave a stage |
|---|---|---|---|---|---|
| (152, 136) | `(start)` | `0x2A` | 4 | 39.0625 | yes |
| (116, 126) | `LEFT` | `0x28` | 0 | 39.0625 | yes |
| (104, 120) | `LEFT+UP` | `0x28` | 0 | 39.0625 | yes |
| (68, 135) | `LEFT+UP+LEFT` | `0x29` | 3 | 15.8125 | yes |
| (141, 135) | `LEFT+UP+RIGHT` | `0x2A` | 4 | 39.0625 | yes |
| (56, 136) | `LEFT+UP+LEFT+UP` | `0x29` | 3 | 15.8125 | yes |
| (93, 126) | `LEFT+UP+LEFT+UP+RIGHT` | `0x28` | 0 | 39.0625 | yes |

7 of those positions hand over a session the player can walk in, and they cover 3 level numbers: `0x28`, `0x29`, `0x2A`. Two cursor positions can be the same stage - the survey found `0x2A` at both ends of one step - which is why the stages below are keyed by the translevel byte and not by the node.

**Writing the cursor is not a teleport.** The first version of the survey parked the cursor by writing `$7E:1F17`/`$7E:1F19` and then pressed A, exactly as 10.36.3's `warp_on_map` does. Three different written positions entered the same level, `0x2A`, every time: the engine keeps its own node index and those two bytes are what it reports, not what it decides. 10.36.3 never noticed because the position it wrote is the position its own walk had already parked the cursor on. The survey now replays a recorded direction sequence from the saved map state before every probe, and reads the cursor back to prove the probe was standing where it meant to.

**How wide a stage is, read off the buffer.** Whether a run that stops has finished is a question about the level, and the level answers it: the buffer at `$7E:C800` stores one subscreen after another, so the last one holding anything but air is where the terrain ends. Yoshi's Island 1 holds 20 subscreens - 5120 px - which is 2.69 times the distance 10.37 calls the goal. That number is what turned the clearance question around: the first campaign gave the controller 3000 frames per attempt and its furthest run covered 4340 px of a 5120 px level, so the wall it hit was the frame budget, not the end of the stage - and the same controller, flown with room enough to finish, does finish it below. The stages are 5120 px, 256 px, 5120 px, 5376 px wide in the order they are listed, and the progress bar on every dashboard is now scaled to its own stage's terrain rather than to a goal distance nobody has seen the engine confirm.

**What a clear is - three definitions in one day.** A stage that ends is a stage the engine leaves, so `scripts/unlock_world_map.py` flies the published clearance controller at the entered level and samples `$7E:0100` every frame instead of trusting a distance. The first definition this file used was 10.37's: `x >= 1900 px`. The second was this script's own first draft - reaching the map mode `0x0E` counts - and it reported every round as cleared, because a game over also ends on the map. The third is the byte: a death leaves `0x14` for `0x0B`, the death animation, and a run that reaches the map without passing through it has finished the stage. That is a discriminator the console supplies, not an interpretation the harness brings.

| round | stage | attempt | px | frames | ended in | died | the map at | cleared |
|---|---|---|---|---|---|---|---|---|
| 0 | `0x2A` at (152, 136) | 0 | 5010.6250 | 4375 | `0x0C` | no | [152, 104] | **yes** |
| 1 | `0x27` at (152, 104) | 0 | 376.5000 | 441 | `0x0B` | yes | [152, 104] | no |
| 1 | `0x27` at (152, 104) | 1 | 355.8125 | 385 | `0x0B` | yes | [152, 104] | no |

1 of 2 rounds cleared. The stage `0x2A` at node (152, 136) - the one 10.36.3 built the savestate for - was finished by attempt 0 of round 0: 5010.6250 px in 4375 frames, the mode going `0x14` to `0x0C` with no `0x0B` anywhere, and the map settling at node (152, 104) - a node the fresh file cannot reach. Re-flown from the same savestate with the same seed, the run reached 5010.6250 px, a difference of 0.000000 px: the clear is reproducible, not one lucky draw. The map then reached 2 positions it did not reach before ((152, 104), (115, 126)), and the stage behind the new node is `0x27`, at node (152, 104), 5376 px of terrain.

**One episode per session, flown by the agent that can conclude it.** `scripts/record_level_dashboards.py` records the same channels 10.60 records at each session the map hands over - surveyed or unlocked - plus the two channels 10.68 added: the full 16-bit tile ids of the window and every live sprite slot. Yoshi's Island 1 is flown by the published clearance controller on its seed-7 draw until the engine itself ends the stage; the blank house session has no goal tape at all and is concluded by walking back out through its door - the left-edge exit the engine performs on its own; the other two sessions are documented by the distilled actor for a fixed horizon, because 10.68 measured no agent concludes them at all. Every episode keeps recording past the verdict with the controls released, so each video shows the engine's own exit sequence - the mode march and the map settling - instead of freezing on the flipped byte.

| stage | translevel | terrain the buffer holds | rightward walk reaches | flown by | the episode | share | the engine's verdict |
|---|---|---|---|---|---|---|---|
| `Yoshi's Island 1` | `0x2A` | 5120 px (20 subscreens) | 974.1250 px (stall) | `mpc` | 5010.6250 px in 4694 frames | 97.86% | **cleared** (left `0x14` for `0x0C`, no `0x0B`) |
| `Yoshi's House` | `0x28` | 256 px (1 subscreen) | 126.1875 px (stall) | `terrain-runner` | 65404.6250 px in 391 frames | 25548.68% | **concluded** (out the door, `0x0E`, no life lost) |
| `Yoshi's Island 2` | `0x29` | 5120 px (20 subscreens) | 412.5625 px (stall) | `dagger` | 406.4375 px in 548 frames | 7.94% | no (the engine ended the session in the death mode and took a life) |
| `Yoshi's Island 3` | `0x27` | 5376 px (21 subscreens) | 369.0000 px (death) | `dagger` | 372.8750 px in 592 frames | 6.94% | no (the engine ended the session in the death mode and took a life) |

The verdict column is the engine's own, and the lives counter completes it: a session is concluded when the engine returned the player to the map without taking a life. The mode byte alone cannot say that - the house's walk out the door marches through `0x0B`, the byte the death animation owns, and the byte would call an exit a death - so the lives counter (`$7E:0DBE`) settles it: an exit leaves it put, a death decrements it. No distance proxy appears in that column, on purpose.

**What the names in the tables and on the panels are worth.** `0x28` Yoshi's House, `0x2a` Yoshi's Island 1 are identified by measurements this repository made: 10.36.3 walked the map from the node a fresh file starts on and read `0x2A` there, and 10.66/10.68 measured `0x28` as the session with no timer, one subscreen of terrain and a door the engine walks the player out through. `0x25` #1 Iggy's Castle, `0x26` Yoshi's Island 4, `0x27` Yoshi's Island 3, `0x29` Yoshi's Island 2 are not: they are the community's translevel table (SMWCentral thread #23605, the Lunar Magic vanilla list, the `snesrev/smw` decompilation) shifted to agree with the anchor, and that table is not consistent with it - it names `0x29` Yoshi's Island 1 and `0x2A` Yoshi's Island 2, while the map position and the game's own progression make `0x2A` the first stage. The measurement that would settle the question was run and is unavailable on this console route: entering the known `0x2A` draws no title card under any of four attempts - from the restored map state and from a genuine power-on boot walked through the menus, each with a pulsed A and with a single deliberate press. Capturing every frame of the load sequence shows the null is blind rather than informative: modes `0x10` and `0x12` are the black fade and `0x13` already holds the drawn level, HUD up and timer at 400 counting, so the instrument never sees the case whose answer is known and cannot clear the cases whose answers are not. `results/dashboard_video_metrics.json` carries `protocol.level_name_source` per render, and `tests/test_dashboard_pipeline.py` refuses a row that claims to be measured when it is not.

![The dashboard on Yoshi's Island 1: the console's own picture, the tile window that stage's models read, and what the actor believed on every frame](../../results/figures/dashboard_level_1.gif)

**Level 1: Yoshi's Island 1.** 4694 frames, 5009.3125 px of progress, the tile window's empty class on 91.16% of its cells, 4.87 MB GIF / 4.90 MB MP4: `results/figures/dashboard_level_1.mp4`.

![The dashboard on Yoshi's House: the console's own picture, the tile window that stage's models read, and what the actor believed on every frame](../../results/figures/dashboard_level_2.gif)

**Level 2: Yoshi's House.** 391 frames, 65406.8125 px of progress, the tile window's empty class on 98.93% of its cells, 0.26 MB GIF / 0.47 MB MP4: `results/figures/dashboard_level_2.mp4`.

![The dashboard on Yoshi's Island 2: the console's own picture, the tile window that stage's models read, and what the actor believed on every frame](../../results/figures/dashboard_level_3.gif)

**Level 3: Yoshi's Island 2.** 548 frames, 406.4375 px of progress, the tile window's empty class on 92.81% of its cells, 1.45 MB GIF / 1.22 MB MP4: `results/figures/dashboard_level_3.mp4`.

![The dashboard on Yoshi's Island 3: the console's own picture, the tile window that stage's models read, and what the actor believed on every frame](../../results/figures/dashboard_level_4.gif)

**Level 4: Yoshi's Island 3.** 592 frames, 370.6250 px of progress, the tile window's empty class on 94.74% of its cells, 1.33 MB GIF / 1.12 MB MP4: `results/figures/dashboard_level_4.mp4`.
