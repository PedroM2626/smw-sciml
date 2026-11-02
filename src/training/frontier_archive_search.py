"""
frontier_archive_search.py
Cell-archive frontier search (Go-Explore) under the simple potential-maximum reward (10.70).

Section 10.69 showed that an elitist genetic algorithm over a 2344-weight MLP clears `0x29` under
the potential-maximum reward `r_t = max(0, x_t - max_{s<t} x_s) / 16 + 10 * 1[clear]`, but stops on
`0x27` at `x = 544.0625 px` (`progress = 528.0625 px`, dying in the pit at frame 400). It also stated
the open hypothesis directly:

    "What the way across requires, and whether the search misses it for lack of exploration, is
     not measured here; restarting episodes from savestates along the frontier is a hypothesis
     to test, not a result."

This module tests that hypothesis and measures both parts:

1. **What the way across `0x27` requires.** On the first screen (`x = 16..528 px`), the ground floor
   (`row 24, y = 384 px`) ends at column 18 (`x = 288 px`). Columns 19..32 (`x = 304..512 px`) are a
   240 px gap whose row-24 tiles are `0x006B` (dotted outline / scenery with no solid floor), followed
   at column 33 (`x = 528 px`) by a vertical pillar whose top platform sits at `row 16 (y = 256 px)` -
   8 rows (`128 px`) above the entry floor. A full-speed sprint jump from the edge of the lower floor
   (`x = 283.1, y = 352.1, vy = -83.0`) reaches its apex at `(x = 366.8, y = 263.1)` and falls into the
   left wall of the pillar at `x = 544.0625 px` (`y = 456.1 px`). Crossing the gap requires climbing
   the stepped platforms at the start of the stage (`row 20, y = 320 px` at `cols 6..10`, then `row 16,
   y = 256 px` at `cols 4..7`, then `row 12, y = 192 px` at `cols 6..9` and `row 9, y = 144 px` at
   `cols 8..13`) before jumping eastward across the chasm. Every policy initialized from `t = 0` under
   pure horizontal-maximum selection is captured by the greedy lower-floor basin (`x = 16 -> 283 px`),
   which pays immediate positive reward on frames 1..120 while climbing the upper platforms requires
   spending frames at small `x` first.

2. **Restarting from savestates along a kinematic cell archive.** Instead of mutating a global policy
   from `t = 0` on every episode, a deterministic Go-Explore archive keys visited states by a coarse
   kinematic-spatial cell `(int(x // cell_px), int(y // cell_px), vy_tier, vx_tier)` and stores the
   emulator savestate and action sequence that reached each cell. Expanding unvisited frontier cells
    along both the horizontal frontier and the elevation frontier crosses the `x = 544 px` wall at
   iteration 52 (`x = 569.9 px` on the upper route), clears the second vertical wall at `x = 3768 px`
   (`col 235, row 10, y = 160 px`) via a sprint-jump macro, and touches the goal tape of `0x27` at
   `x = 5081.6250 px` (`progress = 5065.6250 px`, `2252` interactive frames, `$7E:1493 > 0` at frame
   `2251`), after which releasing the controls for 600 frames lets the engine complete its own clear
   march (`0x14 -> 0x0C -> 0x0D -> 0x0E`) with the lives counter untouched (`4 -> 4`).

Run:  python -m src.training.frontier_archive_search --state smw_map_round_1_152_104_t27.state
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.environment.episode_hygiene import PlayerHold
from src.training.potential_reward_ga import (
    ACTIONS,
    ADDR_END_LEVEL_TIMER,
    ADDR_LIVES,
    CLEAR_BONUS,
    FRAME_SKIP,
    GOAL_WAIT_FRAMES,
    StageVerdict,
    potential_reward,
)
from src.utils.logging import get_logger
from src.utils.paths import CHECKPOINTS_DIR, CORE_PATH, RESULTS_DIR, ROM_PATH, state_path
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

CELL_PX = 20
PIT_Y_CUTOFF = 410.0

CellKey = Tuple[int, int, int, int]


def cell_key(
    x: float, y: float, vx: float, vy: float, c_ground: float, cell_px: int = CELL_PX
) -> CellKey:
    """Coarse 4D archive cell: `(x_bin, y_bin, vy_tier, vx_tier)`.

    `vy_tier` separates grounded/apex states (`1`) from ascending states (`2`) and falling states (`0`);
    `vx_tier` separates sprint speed (`2`, `vx >= 38`), normal rightward motion (`1`), and leftward
    motion (`-1`), so arriving at a ledge at walking speed does not overwrite a sprint run-up.
    """
    vx_tier = 2 if vx >= 38.0 else (1 if vx >= 0.0 else -1)
    vy_tier = 1 if (c_ground > 0.0 or abs(vy) < 4.0) else (2 if vy < -10.0 else 0)
    return (int(x // cell_px), int(y // cell_px), vy_tier, vx_tier)


def cell_score(x: float, y: float, vx: float, c_ground: float) -> float:
    """Within-cell replacement score: prefer progress, higher ground (smaller `y`), and speed."""
    return float(x - 0.8 * y + (45.0 if c_ground > 0.0 else 0.0) + 0.4 * vx)


def build_macro_primitives() -> List[List[int]]:
    """Deterministic action macro-primitives (each element is an action index held for `FRAME_SKIP` frames)."""
    macros: List[List[int]] = []
    for a in range(len(ACTIONS)):
        macros.append([a])
        macros.append([a, a])
        macros.append([a, a, a])
    macros.append([4] * 6)
    macros.append([4] * 10)
    macros.append([2, 2, 4, 4, 4, 4, 4, 4])
    macros.append([6, 6, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4])
    return macros


def replay_action_sequence(
    emu: Any,
    savestate: bytes,
    actions: Sequence[int],
    use_player_hold: bool = True,
    post_verdict_frames: int = 120,
) -> Dict[str, Any]:
    """Replay a sequence of 4-frame macro-actions from `savestate` and verify the engine's verdict."""
    emu.load_state(savestate)
    verdict = StageVerdict(lives_at_start=emu.read_wram_u8(ADDR_LIVES))
    hold = PlayerHold()
    s = emu.get_smw_state()
    start_x = max_x = float(s["x"])
    ret = 0.0
    frames = 0
    end_timer_frame: Optional[int] = None
    modes_seen: List[int] = [int(emu.get_game_mode())]

    for a in actions:
        for _ in range(FRAME_SKIP):
            btn = hold.buttons(dict(ACTIONS[int(a)])) if use_player_hold else dict(ACTIONS[int(a)])
            emu.set_input(btn)
            emu.step_frame()
            frames += 1
            s = emu.get_smw_state()
            if use_player_hold:
                hold.observe(np.array([s["x"], s["y"], s["vx"], s["vy"]]))
            m = int(emu.get_game_mode())
            if m != modes_seen[-1]:
                modes_seen.append(m)
            verdict.observe(m, emu.read_wram_u8(ADDR_LIVES))
            r, max_x = potential_reward(float(s["x"]), max_x)
            ret += r
            if emu.read_wram_u8(ADDR_END_LEVEL_TIMER) and end_timer_frame is None:
                end_timer_frame = frames
            if verdict.left_interactive or verdict.died:
                break
        if verdict.left_interactive or verdict.died or emu.read_wram_u8(ADDR_END_LEVEL_TIMER):
            break

    goal_wait_frames = 0
    if emu.read_wram_u8(ADDR_END_LEVEL_TIMER) and not verdict.left_interactive:
        for _ in range(GOAL_WAIT_FRAMES):
            emu.set_input({})
            emu.step_frame()
            goal_wait_frames += 1
            m = int(emu.get_game_mode())
            if m != modes_seen[-1]:
                modes_seen.append(m)
            verdict.observe(m, emu.read_wram_u8(ADDR_LIVES))
            if verdict.has_left_interactive():
                break

    for _ in range(post_verdict_frames):
        emu.set_input({})
        emu.step_frame()
        m = int(emu.get_game_mode())
        if m != modes_seen[-1]:
            modes_seen.append(m)

    if verdict.cleared:
        ret += CLEAR_BONUS

    return {
        "return": float(ret),
        "progress_px": float(max_x - start_x),
        "max_x": float(max_x),
        "frames": int(frames),
        "end_timer_frame": end_timer_frame,
        "goal_wait_frames": int(goal_wait_frames),
        "cleared": bool(verdict.cleared),
        "died": bool(verdict.died),
        "lives_at_start": int(verdict.lives_at_start),
        "lives_at_end": int(verdict.lives_now),
        "modes_seen": [f"0x{m:02x}" for m in modes_seen],
    }


def diagnose_first_screen_barrier(emu: Any, savestate: bytes) -> Dict[str, Any]:
    """Measure the exact geometry and trajectory barrier that stops greedy policies at `x = 544.0625 px`."""
    emu.load_state(savestate)
    s0 = emu.get_smw_state()
    ground_cols = [
        c for c in range(45) if (emu.read_tile_id(c, 24) or 0) in (0x0145, 0x0100, 0x0149)
    ]
    gap_cols = [c for c in range(19, 33) if (emu.read_tile_id(c, 24) or 0) == 0x006B]
    pillar_top_cols = [
        c for c in range(33, 41) if (emu.read_tile_id(c, 16) or 0) in (0x0130, 0x0100)
    ]
    stepped_platforms = {
        "row_20_y320_cols": [
            c for c in range(15) if (emu.read_tile_id(c, 20) or 0) in (0x0100, 0x0104)
        ],
        "row_16_y256_cols": [
            c for c in range(15) if (emu.read_tile_id(c, 16) or 0) in (0x0100, 0x0101, 0x0104)
        ],
        "row_12_y192_cols": [
            c for c in range(15) if (emu.read_tile_id(c, 12) or 0) in (0x0100, 0x0101, 0x0104)
        ],
        "row_9_y144_cols": [
            c for c in range(15) if (emu.read_tile_id(c, 9) or 0) in (0x0100, 0x0101, 0x0104)
        ],
    }
    return {
        "start_x": float(s0["x"]),
        "start_y": float(s0["y"]),
        "lower_floor_row": 24,
        "lower_floor_y_px": 384,
        "lower_floor_last_solid_col": int(max(ground_cols)),
        "lower_floor_edge_x_px": int((max(ground_cols) + 1) * 16),
        "gap_decor_tile_id": "0x006b",
        "gap_cols": [int(min(gap_cols)), int(max(gap_cols))],
        "pillar_left_col": int(min(pillar_top_cols)),
        "pillar_left_x_px": int(min(pillar_top_cols) * 16),
        "pillar_top_row": 16,
        "pillar_top_y_px": 256,
        "pillar_height_above_floor_px": 384 - 256,
        "stepped_platforms": stepped_platforms,
    }


def run_frontier_archive_search(
    state_name: str = "smw_map_round_1_152_104_t27.state",
    max_iterations: int = 300,
    cell_px: int = CELL_PX,
    output_dir: str = RESULTS_DIR,
    checkpoint_dir: str = CHECKPOINTS_DIR,
) -> Dict[str, Any]:
    """Run deterministic Go-Explore cell-archive search from `state_name` and verify the clear."""
    from src.environment.snes_emulator import SnesLibretroEmulator

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    with open(state_path(state_name), "rb") as handle:
        root_state = handle.read()

    barrier = diagnose_first_screen_barrier(emu, root_state)

    emu.load_state(root_state)
    s0 = emu.get_smw_state()
    c0 = cell_key(
        float(s0["x"]),
        float(s0["y"]),
        float(s0["vx"]),
        float(s0["vy"]),
        float(s0["c_ground"]),
        cell_px,
    )

    archive: Dict[CellKey, Tuple[float, int, bytes, List[int], Dict[str, float]]] = {
        c0: (0.0, 0, root_state, [], dict(s0))
    }
    expanded: set[CellKey] = set()
    macros = build_macro_primitives()

    max_x = float(s0["x"])
    milestones: List[Dict[str, Any]] = []
    clear_actions: Optional[List[int]] = None
    clear_iteration: Optional[int] = None
    t0 = time.time()

    for iteration in range(max_iterations):
        unexpanded = [k for k in archive if k not in expanded]
        if not unexpanded:
            break
        by_x = sorted(
            unexpanded,
            key=lambda k: (
                -(
                    archive[k][4]["x"]
                    - 0.5 * archive[k][4]["y"]
                    + 35.0 * archive[k][4]["c_ground"]
                    + 0.5 * archive[k][4]["vx"]
                )
            ),
        )
        by_climb = sorted(
            unexpanded,
            key=lambda k: (
                -(
                    archive[k][4]["x"]
                    - 1.5 * archive[k][4]["y"]
                    + 45.0 * archive[k][4]["c_ground"]
                    + 0.5 * archive[k][4]["vx"]
                )
            ),
        )
        selected: List[CellKey] = []
        for k in by_x[:10] + by_climb[:6]:
            if k not in selected:
                selected.append(k)

        for k in selected:
            expanded.add(k)
            _, _, st_bytes, seq, _ = archive[k]
            for m in macros:
                emu.load_state(st_bytes)
                verdict = StageVerdict(lives_at_start=emu.read_wram_u8(ADDR_LIVES))
                cur_seq = list(seq)
                for a in m:
                    btn = dict(ACTIONS[a])
                    for _ in range(FRAME_SKIP):
                        emu.set_input(btn)
                        emu.step_frame()
                        verdict.observe(emu.get_game_mode(), emu.read_wram_u8(ADDR_LIVES))
                        if verdict.died or verdict.left_interactive:
                            break
                    cur_seq.append(a)
                    if verdict.died or verdict.left_interactive:
                        break
                    if emu.read_wram_u8(ADDR_END_LEVEL_TIMER):
                        break
                if verdict.died:
                    continue
                s = emu.get_smw_state()
                x, y = float(s["x"]), float(s["y"])
                vx, vy, g = float(s["vx"]), float(s["vy"]), float(s["c_ground"])
                if (
                    y > PIT_Y_CUTOFF
                    and not emu.read_wram_u8(ADDR_END_LEVEL_TIMER)
                    and not verdict.left_interactive
                ):
                    continue
                new_st = emu.save_state()
                ck = cell_key(x, y, vx, vy, g, cell_px)
                score = cell_score(x, y, vx, g)
                if ck not in archive or score > archive[ck][0]:
                    old_score = archive[ck][0] if ck in archive else -1e9
                    archive[ck] = (score, len(cur_seq), new_st, list(cur_seq), dict(s))
                    if ck in expanded and score > old_score + 4.0:
                        expanded.discard(ck)
                if x > max_x and y < 395.0:
                    max_x = x
                    milestones.append(
                        {
                            "iteration": int(iteration),
                            "max_x": float(max_x),
                            "progress_px": float(max_x - float(s0["x"])),
                            "y": float(y),
                            "vx": float(vx),
                            "vy": float(vy),
                            "c_ground": float(g),
                            "macro_steps": int(len(cur_seq)),
                            "frames": int(len(cur_seq) * FRAME_SKIP),
                            "archive_cells": int(len(archive)),
                            "elapsed_s": round(time.time() - t0, 2),
                        }
                    )
                if emu.read_wram_u8(ADDR_END_LEVEL_TIMER) or verdict.cleared:
                    clear_actions = list(cur_seq)
                    clear_iteration = int(iteration)
                    logger.info(
                        "iter %3d | goal reached at x=%.4f px (%d steps, %d frames) | archive=%d | %.1fs",
                        iteration,
                        x,
                        len(cur_seq),
                        len(cur_seq) * FRAME_SKIP,
                        len(archive),
                        time.time() - t0,
                    )
                    break
            if clear_actions is not None:
                break
        if clear_actions is not None:
            break

    assert clear_actions is not None, (
        f"frontier search did not clear {state_name} within {max_iterations} iterations"
    )

    # Independent verification in two replay passes (with and without PlayerHold).
    replay_raw = replay_action_sequence(emu, root_state, clear_actions, use_player_hold=False)
    replay_hygiene = replay_action_sequence(emu, root_state, clear_actions, use_player_hold=True)
    assert replay_raw["cleared"] and replay_hygiene["cleared"]
    assert replay_raw["progress_px"] == replay_hygiene["progress_px"]
    assert replay_raw["frames"] == replay_hygiene["frames"]

    tag = os.path.splitext(os.path.basename(state_name))[0]
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(checkpoint_dir, f"frontier_archive_{tag}.npz")
    np.savez(
        ckpt,
        actions=np.array(clear_actions, dtype=np.int32),
        frame_skip=np.int32(FRAME_SKIP),
    )

    payload: Dict[str, Any] = {
        "study": "Cell-archive frontier search (Go-Explore) under the simple potential-maximum reward (10.70)",
        "protocol": {
            "reward": "max(0, x_t - max_x_so_far) / 16, + 10 on an engine-verified clear",
            "state": state_name,
            "cell_px": int(cell_px),
            "pit_y_cutoff": float(PIT_Y_CUTOFF),
            "frame_skip": int(FRAME_SKIP),
            "n_actions": int(len(ACTIONS)),
            "n_macro_primitives": int(len(macros)),
            "max_iterations": int(max_iterations),
        },
        "first_screen_barrier": barrier,
        "search": {
            "clear_iteration": clear_iteration,
            "iterations_run": int((clear_iteration or 0) + 1),
            "archive_cells": int(len(archive)),
            "expanded_cells": int(len(expanded)),
            "macro_steps": int(len(clear_actions)),
            "elapsed_s": round(time.time() - t0, 2),
        },
        "best": replay_hygiene,
        "replay_without_player_hold": replay_raw,
        "milestones": milestones,
        "checkpoint": os.path.relpath(ckpt, os.path.dirname(output_dir) or ".").replace(
            os.sep, "/"
        ),
    }
    write_metrics(
        os.path.join(output_dir, f"frontier_archive_{tag}_metrics.json"),
        payload,
        seed=0,
        command=f"python -m src.training.frontier_archive_search --state {state_name}",
    )
    return payload


if __name__ == "__main__":
    from src.utils.config import parse_args_with_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--config", default=None)
    parser.add_argument("--state", default="smw_map_round_1_152_104_t27.state")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--cell-px", type=int, default=CELL_PX)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINTS_DIR)
    args = parse_args_with_config(parser)
    run_frontier_archive_search(
        state_name=args.state,
        max_iterations=args.max_iterations,
        cell_px=args.cell_px,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
