[← 10.35](10-35-formal-learning-curves-spatial-holdout-ood-with-danger.md) · [Contents](../../README.md#contents) · [10.37 →](10-37-analytical-and-oracle-model-baselines.md)

### 10.36 Yoshi's Island 2 Capture: Blocked with Full Diagnostics

`python -m scripts.navigate_to_level --level 2` reaches *a* level entry (mode 0x14)
but the post-entry story message box never reaches a playable handoff despite
an instrumented campaign (frame-capture debugging via the pixel API):
B/A/X holds and pulses, START hold, Y hold, single Y edge (fires a 0x14→0xC
transition that returns to the map, 0xE), 1500-frame idle waits. Findings
locked into the script, which **raises instead of saving garbage**:
message dismissal needs a Y *edge* (consistent with the `$7E:0016` latch);
`set_input` maps START/SELECT/L/R (previously silently dropped) and raises
`KeyError` on unknown buttons; and a savestate is now refused whenever the engine
never reaches interactive mode, instead of writing a file-select frame as if it
were a level start.

#### 10.36.1 Reproduced Attempt (2026-09-22, machine-readable evidence)

The attempt is no longer only prose: `results/yi2_capture_attempt.json` records it
with full `_meta` provenance.

| Stage | Observed | Expected for a playable state |
| :--- | :--- | :--- |
| World-map traversal | YI2 node entered on map round **4** (past the Yoshi's House node) | - |
| Level-entry transition | `$7E:0100 = 0x14` reached on **frame 0** of the wait | mode 0x14 |
| Coordinates at entry | $X = 136$, $Y = 313$ | near a level spawn |
| Animation state at entry | `$7E:0072 = 11` (`0x0B`, level-entry slide) | $\in \{0, 1, 2\}$ |
| After 12 Y-pulse dismissal attempts | $X = 13.0$, $Y = 65{,}502$ (i.e. $\texttt{0xFFFE}$ read unsigned) | stable, plausible |
| Stability gate | **0** of 60 consecutive plausible frames (1200 frames sampled) | 60 |
| Outcome | `blocked: WRAM never stabilised after the message box` → **no savestate written** | - |

Interpretation kept honest: the $0x0B$ animation state at entry plus the wrapped
$Y$ reading mean the engine is mid-transition, and the byte-level evidence is
consistent with the earlier suspicion of **entry-point ambiguity** - the $X = 13$
reached after dismissal is a House-like spawn, not the $X = 16$ of Yoshi's Island 1
or a YI2 start. The open question, the recipe and the gates (60 stable plausible
frames + movement $\Delta X \gt 10$ px) are documented here and in the script so the
next attempt starts from evidence, not guesses.

#### 10.36.2 Corrected Harness - Bug Fix and a Stronger Integrity Gate (still blocked)

Revisiting the capture surfaced two genuine harness defects and refined the
root cause, without changing the honest verdict:

1. **The old Y-pulse dismissal was itself the corrupting action.** Section 10.36
   already recorded that a Y *edge* on the entry fires a `0x14 -> 0x0C` map
   return, yet the script still pulsed Y to "close the message box". That pulse
   is what bounced the state into the wrapped $Y = 65{,}502$ read. Removing it
   (idle-settle instead) lets the level-entry slide complete on its own and
   reaches a deep, plausible in-level state ($X \approx 137.7$, $Y \approx 301.8$)
   rather than a House-like $X = 13$.
2. **A static stability gate can be fooled by a frozen frame.** The
   stable-but-not-playable state passes a 60-consecutive-plausible-frames check
   because the WRAM reads are simply *not advancing*. The gate was upgraded to a
   **control-response probe**: Mario must actually move under a held RIGHT
   ($\Delta X \gt 6$ px), with START toggles to break a possible entry fade/pause.

Under the corrected harness the capture is **still blocked**, but now with a
precise cause: the player's physics do not step at this node at all -
$X = 137.6875$, $v_y = 10.0$ and animation byte $0x24$ are byte-identical across
repeated runs even under sustained RIGHT + START, and $v_y$ never grows toward
terminal velocity while ungrounded. This is an engine-level *handoff* failure
(the world-map traversal is not landing on a controllable YI2 entry), not a
wait/timing issue the script can paper over. The capture therefore writes **no**
savestate and `results/yi2_capture_attempt.json` is refreshed to
`blocked: no control handoff (right dx=0.0, air=36)`. Fabricating a handoff from
a frozen frame is explicitly refused - the same integrity rule that keeps every
other blocked result honest. The OOD-with-danger holdout that was still deferred when this paragraph was written has since been delivered, as Section 10.35.

#### 10.36.3 The Same Gate Applied to Yoshi's Island 1 (2026-10-11): the Old Schedule Was the Attract Demo

The control-response gate above was written for stage 2, and stage 1 had been saving its savestate
with no behavioural verification at all. Applying the gate to `--level 1` made it raise, and chasing
why turned up a route that works - so this section records both, with the A/B in
`results/level_entry_verification.json` (`python scripts/verify_level_entry.py`).

**What the old schedule actually reaches.** Polled frame by frame, the schedule that holds START for
20 frames and then B walks the mode byte `0x08 -> 0x0B -> 0x11 -> 0x13 -> 0x14`, the PPU draws a real
level with the welcome text box (confirmed by exporting the frame), and Mario sits at $X = 128$:
0.0 px of travel over 400 frames of RIGHT + dash + jump, and **0.00% of consecutive frame pairs
translate at all**. With no input whatsoever the mode stops at `0x07`, so the walk to `0x14` is
caused by the presses - it is just the wrong gesture: this is the title screen's attract demo, a
rendered level played by recorded input, and the controller is not connected to it.

**What reaches a playable level.** The route is the one documented by the MIT-licensed
[araujoviana/burn-mario-rl](https://github.com/araujoviana/burn-mario-rl) (`src/env.rs`,
`src/ram.rs`): sit out the black intro, pulse **START with release gaps** until the *world map*
(`0x0E`) is live - a mode the old schedule never waits for - enter Yoshi's House with **A** rather
than B, walk back out, and then write the map cursor `$7E:1F17/$7E:1F19` and press A, because the
game derives the level from wherever the cursor is on the frame the button is read; writing the
translevel byte does nothing. Ported into `navigate_to_level.py`, that reaches the node at cursor
(152,136), translevel `0x2A`, and the A/B measures the difference:

| Arm | Final mode | x over 400 frames | Camera travel | Pairs of frames that translate |
| :--- | :---: | :---: | :---: | :---: |
| old schedule | `0x14` | 128.0 -> 128.0 (0.0 px) | 0 -> 0 | 0.00% |
| measured route | `0x14` | 16.0 -> 882.8 (866.8 px) | 0 -> 774 | 71.93% |

So the emulator's own frames *can* show this player in this level, and `data/raw/
smw_yoshi_island_1_level.state` is the verified state for it - saved at $X = 16$, the coordinate every
committed recording starts from, after the gate below.

**Two bugs the gate caught in the code that wrote it.** First, the plausibility check vetoed
animation byte 0, inherited from the stage-2 investigation; a Mario standing on solid ground in a
level that responds to input reads exactly 0, so the byte is now recorded and no longer decides
anything - the behavioural probe is the evidence. Second, the savestate was being taken *after* the
verification walked Mario 193 px down the stage, so the "level start" it produced was not the level
start; the state is now captured before the probes and restored afterwards.

**What this does not settle.** The genuine level's terrain does not match the terrain the committed
recordings walked. The figure this paragraph used to carry came from a scratch run with no artifact
behind it and a sampling rule it never defined, so it is withdrawn rather than restated; the
measurement is redone in §10.61, which defines the instrument, applies it to the recording and to
both sessions live, and separates the two explanations it could not; §10.64 then names the buffer. What remains true here is the
shape of the problem: an open question this repository had been assuming away - which level the
forced-mode session, the one every telemetry file here comes from, actually has loaded in its
`$7E:C800` buffer. The physics constants identified from those files are the engine's (16.000001
sub-pixels per pixel, gravity +3.0/+6.0), which is what every study in sections 10.37 onward
measures; the *stage* was unresolved until 10.61, and §10.31.1's frames are the menu either way.

---
