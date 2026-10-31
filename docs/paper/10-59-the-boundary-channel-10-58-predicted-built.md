[← 10.58](10-58-the-residue-of-section-4-1-is.md) · [Contents](../../README.md#contents) · [10.60 →](10-60-the-dashboard-at-the-top-of-this.md)

### 10.59 The Boundary Channel 10.58 Predicted, Built, and Refuted

10.58 read the exceptions to §4.1 as a constraint the state vector does not carry, and predicted that a ninth channel - the boundary, or the camera offset - would remove them. Testing that needed an address the repository has never had, so the channel was built first: `scripts/scan_scroll_address.py` dumps all 128 KB of WRAM on each of 541 frames of a scripted run, ranks every 16-bit word by how well it follows Mario's x, and keeps the ones that satisfy the axioms of a layer scroll. Then `scripts/record_boundary_gameplay.py` recorded 13,951 transitions with that channel, the engine mode byte, and a CRC of all of WRAM beside the published 8D state.

**The instrument.** The address was identified, not assumed:

| Quantity | Measured |
| :--- | :--- |
| Camera address | `$7E:001A` |
| 16-bit words that follow x | 7 |
| Words satisfying the scroll axioms | `$7E:001A`, `$7E:1462` |
| Parallax sibling (exactly half) | `$7E:001E` |
| Frames dumped | 541 |
| Mario's travel | 16 to 944 px |
| Screen position on the run | 16.00 to 144.44 px |

The screen position `x - camera` stays inside the 256 px window on 100.00% of the recording's frames and the camera is never ahead of Mario, so the channel behaves; the words that pass the axioms are the chosen one and its mirror, and the word two slots away holds exactly half of it, which is what a parallax layer does.

**The prediction test.** Coverage of the exception frames, as the edge margin moves:

| Margin (px) | Channel fires | Exceptions covered | Live exceptions covered | Lift |
| :--- | :---: | :---: | :---: | :---: |
| 8 | 0.04% | 0.50% | 3.60% | 13.9 |
| 12 | 0.18% | 1.00% | 7.19% | 5.5 |
| 16 | 0.25% | 1.00% | 7.19% | 4.0 |
| 20 | 0.32% | 1.00% | 7.19% | 3.1 |
| 24 | 0.42% | 1.00% | 7.19% | 2.4 |
| 32 | 0.64% | 1.00% | 7.19% | 1.6 |

At the widest margin the channel names 10 of the 1003 exception frames (1.00%), and removing them moves the identity from 92.79% to 92.84% - 0.05 pp of a 7.21 pp deficit. The boundary is not where the residue is.

**What the exception frames actually are.** Every label below is tested in this order, and the counts sum to the exceptions:

| Exception class | Frames | Share of exceptions | The label fires on |
| :--- | :---: | :---: | :---: |
| `paused` | 864 | 86.14% | 6.19% |
| `screen_bound` | 10 | 1.00% | 0.25% |
| `wall_ahead` | 124 | 12.36% | 63.26% |
| `camera_step` | 0 | 0.00% | 41.21% |
| `unexplained` | 5 | 0.50% | - |

On 86.14% of them the *whole player record* repeats: every channel of the next state - position, velocity, all four collision flags - equals the current one, and the residue is exactly minus the velocity because nothing moved. The console did not stop: the CRC over all of WRAM changes on 100.00% of the repeating frames (0 identical), the mode byte reads interactive (0x14) on all 864 of them, and they arrive as 21 runs of 41.1 frames on average (longest 42), in 21 of the 40 episodes. A player object that is not being processed, not a body held by a wall.

**The same test on the recordings 10.58 measured.** They carry no camera and no CRC, but the repeat signature needs neither:

| Recording | Exceptions | Exceptions repeating | Identity exact | Identity with repeats dropped |
| :--- | :---: | :---: | :---: | :---: |
| `gameplay` | 571 | 92.64% | 92.92% | 99.44% |
| `jump` | 301 | 97.01% | 95.07% | 99.85% |
| `multi_entity` | 371 | 74.39% | 98.11% | 99.51% |
| `set_multi_entity` | 151 | 80.79% | 95.77% | 99.16% |
| `sprint` | 478 | 94.56% | 93.40% | 99.62% |
| `tilemap` | 755 | 92.98% | 92.69% | 99.45% |

Findings:

1. **The channel is real and it is not the explanation.** 1.00% of exception frames sit at a window edge, and the identity with them removed is within 0.05 pp of the identity without any channel at all. 10.58's prediction is refuted on its own instrument.
2. **The residue's exceptions are mostly a paused simulation.** 74.39%-97.01% of the exception frames on the six committed recordings repeat their entire player record, and on this recording 100.00% of the repeating frames are exceptions - a repeat *is* an exception, because a body whose velocity byte holds and whose position does not move violates §4.1 by construction.
3. **Section 4.1 is far closer to exact than published.** With the repeats dropped the identity holds on 99.16%-99.85% of the adjacent frames of the six recordings, against the 92.69%-98.11% 10.58 printed, and on 98.93% of this recording's frames against 92.79%. The gap is 6.14 pp here.
4. **What 10.58 called memory is the pause showing up as memory.** Its persistence lift of 104x-626x and its mean runs of 4.16-33.44 frames are the same objects measured twice: once as a conditional probability and once as a run length. The renewal-process kernel that covered the intervals is covering *this* - a deterministic block of repeated frames - and the calibration conclusion that the residue is not noise stands, for a different reason than the one given.
5. **What survives is the one-pixel reposition, and no positional channel explains it.** 139 exception frames here are not repeats, 84.89% of them exactly one pixel off the velocity (residue atoms -16.0, +16.0 sub-pixels carry 118 of them). The terrain ahead of Mario labels 124 of the 1003 exceptions, but it fires on 63.26% of all frames, so it discriminates nothing; the camera steps on 41.21% of frames and labels 0 exceptions.

**What this changes.** The camera channel is committed machinery from here: `wram.ADDR_CAMERA_X` and `SnesLibretroEmulator.get_camera_x()` exist, are gated, and the next recording that wants a boundary can read one. The liveness channels - mode and CRC - are the part the recordings were missing, and any future study that scores exception *rates* on these npz files has to say whether it dropped the repeats: 10.58's numbers were measured as reported, and what is withdrawn is the word *clamp*. The follow-up this makes unavoidable is in the recorder, not the model: end an episode when the player record stops changing, or when the mode leaves 0x14, so the pause is never in the dataset to be learned.

**Limitations.** (i) The pause is identified by its signature, not its cause. The harness restores a savestate that comes up in mode 0x08 and writes 0x14 to force interactive physics (§10.38.1), and a player object that stops integrating for ~41 frames inside that forced session is consistent with the compromise - but this section measured the repeats, the CRC and the mode, and nothing more. (ii) The channel sweep is measured on the one recording that has a camera; the six older recordings can only be tested for the repeat signature, which is what they are shown for. (iii) The terrain label's row set (offsets -1, 0, 1, 2 from Mario's tile row) is a choice; widening it can only raise its fire rate, which is already 63.26%, so no choice of rows makes it a detector. (iv) The residue here is the horizontal identity only; the vertical one has its own ground rule in §4.3.5 and is not scored.

*Regenerate: `make boundary-channel` (or `python -m src.evaluation.boundary_channel_study`). The recording behind it: `python scripts/record_boundary_gameplay.py`; the address behind that: `python scripts/scan_scroll_address.py`.*
