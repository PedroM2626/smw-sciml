"""create_0x26_frontier_archive.py - Generate aquatic 0x26 baseline frontier action sequence.

Produces results/checkpoints/frontier_archive_smw_map_round_2_184_72_t26.npz
used for distilling the initial aquatic PIML stage policy.
"""

from __future__ import annotations

import os

import numpy as np

from src.utils.paths import CHECKPOINTS_DIR


def create_0x26_frontier_archive() -> str:
    """Generate and serialize the baseline action sequence for stage 0x26."""
    actions = [4] * 260 + [7] * 180 + [2] * 25
    actions_arr = np.array(actions, dtype=np.int32)
    out_path = os.path.join(CHECKPOINTS_DIR, "frontier_archive_smw_map_round_2_184_72_t26.npz")
    np.savez_compressed(out_path, actions=actions_arr, frame_skip=4)
    return out_path


if __name__ == "__main__":
    path = create_0x26_frontier_archive()
    print(f"Created aquatic 0x26 frontier archive at {path}")
