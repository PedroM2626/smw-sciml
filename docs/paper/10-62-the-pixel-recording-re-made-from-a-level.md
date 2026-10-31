[← 10.61](10-61-which-level-the-committed-recordings-actually-walk.md) · [Contents](../../README.md#contents) · [10.63 →](10-63-the-held-player-has-two-mechanisms-and.md)

### 10.62 The Pixel Recording, Re-made From a Level That Is Really Entered

10.31.1's finding was a null: the committed RGB frames never translate while the recorded player moves. A null measured on a recording nobody chose is weak evidence - the alignment might simply be blind - so this adds the positive control the old file could not contain: the same instrument, applied to a recording made from the verified level-entry savestate of 10.36.3, with the camera stored alongside the state.

| recording | frames | pairs | picture shifted on | mean |dx| px/frame |
| --- | --- | --- | --- | --- |
| `data/raw/smw_pixel_dataset.npz` (forced-mode, published) | 2,720 | 2,719 | 0.00% | 3.9429 |
| `data/raw/smw_pixel_level_dataset.npz` (level entry, new) | 4,112 | 4,111 | 36.37% | 3.3552 |

The second row's aggregate is not the interesting one, because inside the camera's dead zone the player moves and the scene must not - scoring a pixel recording against the player's own x asks the wrong question. Split by the camera instead, on the new recording's 1,628 pairs where it moved by at least 2 level px:

* the picture holds still on 11.55% of them, and agrees in sign with the camera on 88.39% -
* while on the pairs where the camera did not move, and the player was not one of the 145 frames the engine was holding, the picture holds still on 97.86%.

That is the discrimination the null needed: the same alignment that reports zero shift on 100.00% of the published recording's pairs reports movement on 88.45% of the pairs where a level really is scrolling, and stillness where the camera says there is nothing to scroll. The synthetic control rides along unchanged - a frame slid by 1, 2 and 3 px is recovered exactly on 100.00% of its trials at each depth.

**What this changes, and what it does not.** It confirms 10.31.1's reading rather than replacing it: the committed frames are the menu, and the estimator trained on them was trained on the menu. Those published numbers stand as measurements of the file they name, and retraining on this recording is a separate piece of work, not a consequence of this one. What is new is that the repository now holds a pixel recording of a genuinely entered, genuinely rendered, controllable level - 787.62 px of travel, 688.00 px of camera, 145 held frames the recorder cleared with the operator rule of 10.60 - and an instrument that can tell the two recordings apart. The recording is a new file: `data/raw/smw_pixel_level_dataset.npz`, written by `python scripts/record_pixel_gameplay.py --state data/raw/smw_yoshi_island_1_level.state`, and the published `data/raw/smw_pixel_dataset.npz` is untouched.
