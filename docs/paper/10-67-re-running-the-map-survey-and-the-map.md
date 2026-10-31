[← 10.66](10-66-the-same-dashboard-on-every-stage-the.md) · [Contents](../../README.md#contents) · [10.68 →](10-68-the-tile-map-read-the-way-the.md)

### 10.67 Re-running the Map Survey, and the Map a Clear Opens

10.66 ended on a map that is not the map a fresh file opens: finishing a stage moves the cursor and unlocks a node. `scripts/survey_world_map.py --from-map-state` surveys a saved map rather than booting to one, and pointed at the state the clearance campaign kept after its clear it reaches 8 cursor positions where the published route reaches 7. The level numbers it hands over are `0x27`, `0x28`, `0x29`, `0x2A`. One of them was not on the published map - `0x27`, the stage the clear itself opened - and none left it. Nothing the opened map hands over is a stage that 10.66 had not already listed.

| map cursor | reached by | translevel | timer | terrain the buffer holds | right-press dx | the timer gate calls it a stage |
|---|---|---|---|---|---|---|
| (152, 104) | `(start)` | `0x27` | 3 | 5376 px | 39.0625 | yes |
| (152, 136) | `DOWN` | `0x2A` | 4 | 5120 px | 39.0625 | yes |
| (115, 126) | `DOWN+LEFT` | `0x28` | 0 | 256 px | 39.0625 | no |
| (104, 120) | `DOWN+LEFT+UP` | `0x28` | 0 | 256 px | 39.0625 | no |
| (68, 135) | `DOWN+LEFT+UP+LEFT` | `0x29` | 3 | 5120 px | 17.0625 | yes |
| (141, 135) | `DOWN+LEFT+UP+RIGHT` | `0x2A` | 4 | 5120 px | 39.0625 | yes |
| (56, 136) | `DOWN+LEFT+UP+LEFT+UP` | `0x29` | 3 | 5120 px | 17.0625 | yes |
| (93, 126) | `DOWN+LEFT+UP+LEFT+UP+RIGHT` | `0x28` | 0 | 256 px | 39.0625 | no |

**The node the clear opened was already a stage; the position it reached was not.** The campaign settled the cursor on (152, 104), and that node enters `0x27` with a timer of 3 hundreds and 5376 px of terrain - the stage 10.66 records a dashboard for. The cursor position the published route could not get to is (115, 126): it is walkable, it loads `0x28`, its timer reads 0 and its buffer holds 256 px. That is the session the published route found at (116, 126) - `0x28` there too, timer 0, x = 128.0 px at entry on both - one column east, so it is a path tile of a stage already counted, not a fifth stage. The survey's answer to *how many stages are there* is unchanged by one having been cleared.

**The survey repeats itself, to the byte.** `make survey-map-replay` runs the published route again into its own artifact. Across 7 shared nodes and 14 compared fields it returns 0 differences - the walk of the cursor, the mode sequences, the entry frames, the timers, the entry coordinates and the right-press distances are all exactly what 10.66 published. Each capture also records the SHA-256 of the bytes it wrote, and hashing the files the published survey names matches that digest for 7 of 7: the console state the survey captures at a node is reproduced exactly. The replay wrote its own states under `smw_replay_node` and they are not kept - 0 of them are on disk against the 7 the published survey names, and the opened route keeps 0 of its 8: the replay's bytes are the committed files, and the opened route's are not, which is the next paragraph.

**Same node, different console.** The two routes agree about what a node *does* and disagree about the bytes it *is*: of the 6 cursor positions both entered, 0 have the same state bytes. A savestate carries the route, not just the level - the map flags, the session that arrived. The probe records it twice over: of the 9 field differences between the two surveys, 6 are `path`, which is a property of the map each route started from rather than of the node it ends at, and the other 3 describe the node itself: `right_response_px` at (56, 136), 15.8125 against 17.0625; `right_response_px` at (68, 135), 15.8125 against 17.0625; `entry_frame` at (152, 136), 166 against 167. None of it changes the stage list - every distance on either side of that comparison clears the gate's 6 px threshold - but a number quoted to the quarter pixel would not survive the change of route. That is why every per-stage distance in 10.66 names the savestate it was recorded from.
