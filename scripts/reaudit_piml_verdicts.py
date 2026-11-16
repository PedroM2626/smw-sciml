"""Re-evaluate a committed PIML-PPO checkpoint with the corrected stage verdict and rewrite its block.

`StageVerdict` used to call any exit from mode `0x14` a clear, which includes the fade and load a
level goes through when Mario takes a pipe or a door into a sub-level (`0x0F`-`0x13`). The aquatic
stage `0x26` has such a pipe at x = 2751.6 px, so its evaluation stopped there, recorded 1860 frames
and a "clear" with no end-level timer, and 10.74 read the pipe as the goal tape. With the verdict
corrected, the evaluation keeps going through the pipe; this script re-runs it from the stage's
savestate and rewrites the `closed_loop_evaluation` block of both artifacts that carry it, keeping
the superseded values under `verdict_correction` so the change is on the record.

Needs the core, the ROM and the stage's savestate. Run:

    python scripts/reaudit_piml_verdicts.py --state smw_map_round_2_184_72_t26.state
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.environment.snes_emulator import SnesLibretroEmulator  # noqa: E402
from src.training.piml_stage_ppo import (  # noqa: E402
    PIMLStageActorCritic,
    evaluate_closed_loop_piml,
    hold_trigger_for,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import CHECKPOINTS_DIR, CORE_PATH, RESULTS_DIR, ROM_PATH, state_path  # noqa: E402
from src.utils.provenance import write_metrics  # noqa: E402

logger = get_logger(__name__)

KEEP = (
    "cleared",
    "died",
    "return",
    "progress_px",
    "max_x",
    "frames",
    "macro_steps",
    "end_timer_frame",
    "goal_wait_frames",
    "lives_at_start",
    "lives_at_end",
    "modes_seen",
    "mean_chosen_action_prob",
    "value_first",
    "value_last",
    "mean_executed_physics_violation",
)


def reevaluate(state_name: str) -> Dict[str, Any]:
    tag = os.path.splitext(os.path.basename(state_name))[0]
    blob = torch.load(
        os.path.join(CHECKPOINTS_DIR, f"piml_ppo_{tag}.pt"), map_location="cpu", weights_only=False
    )
    model = PIMLStageActorCritic(
        obs_dim=int(blob["obs_dim"]),
        phys_dim=int(blob["phys_dim"]),
        hidden_dim=int(blob["hidden_dim"]),
        n_actions=int(blob["n_actions"]),
        cbf_beta=float(blob.get("cbf_beta", 0.25)),
    )
    model.load_state_dict(blob["model_state_dict"])
    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    try:
        with open(state_path(state_name), "rb") as handle:
            root_state = handle.read()
        trigger = int(blob.get("hold_trigger", hold_trigger_for(state_name)))
        result = evaluate_closed_loop_piml(
            emu, root_state, model, torch.device("cpu"), hold_trigger=trigger
        )
    finally:
        emu.close()
    return {key: result[key] for key in KEEP}


def rewrite(path: str, locate: Any, state_name: str, fresh: Dict[str, Any], note: str) -> None:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.pop("_meta", None)
    block = locate(payload)["closed_loop_evaluation"]
    superseded = {key: copy.deepcopy(block.get(key)) for key in KEEP}
    block.update(fresh)
    block["verdict_correction"] = {"note": note, "superseded": superseded}
    write_metrics(
        path,
        payload,
        seed=42,
        command=f"python scripts/reaudit_piml_verdicts.py --state {state_name}",
    )
    logger.info("rewrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    fresh = reevaluate(args.state)
    logger.info("re-evaluated %s: %s", args.state, {k: fresh[k] for k in KEEP if k != "modes_seen"})
    note = (
        "Recomputed with the corrected StageVerdict: the sub-level transition (modes 0x0F-0x13) "
        "is no longer counted as leaving the level, so the evaluation continues through it and "
        "ends at the end-level timer and the level-end march (0x0C). The superseded block stopped "
        "at the first transition."
    )
    tag = os.path.splitext(os.path.basename(args.state))[0]
    rewrite(
        os.path.join(RESULTS_DIR, "piml_stage_ppo_metrics.json"),
        lambda payload: payload["stages"][args.state],
        args.state,
        fresh,
        note,
    )
    aquatic = os.path.join(RESULTS_DIR, "piml_stage_ppo_aquatic_0x26_metrics.json")
    if tag.endswith("t26") and os.path.isfile(aquatic):
        rewrite(aquatic, lambda payload: payload["stage"], args.state, fresh, note)


if __name__ == "__main__":
    main()
