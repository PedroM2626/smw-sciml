"""prepare_full_stage_vision_dataset.py - Extract full-stage RGB and state pairs.

Packages data/raw/smw_dashboard_run_smw_map_node_152_136_t2a.npz into
data/raw/smw_pixel_full_stage_t2a.npz for training the extended PixelStateEstimator.
"""

from __future__ import annotations

import os

import numpy as np

from src.utils.paths import DATA_RAW_DIR


def prepare_full_stage_vision_dataset(
    src_file: str = "smw_dashboard_run_smw_map_node_152_136_t2a.npz",
    dst_file: str = "smw_pixel_full_stage_t2a.npz",
) -> str:
    """Load recording run and format into frames and 8D state ground truths."""
    src_path = os.path.join(DATA_RAW_DIR, src_file)
    dst_path = os.path.join(DATA_RAW_DIR, dst_file)
    raw = np.load(src_path)
    rgb = raw["rgb"]
    states = raw["states"][:, :8].astype(np.float32)

    np.savez_compressed(dst_path, frames=rgb, states=states)
    return dst_path


if __name__ == "__main__":
    out = prepare_full_stage_vision_dataset()
    print(
        f"Generated full-stage vision dataset: {out} ({os.path.getsize(out) / (1024 * 1024):.1f} MB)"
    )
