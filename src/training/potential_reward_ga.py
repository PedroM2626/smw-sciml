"""
potential_reward_ga.py
A small policy searched by a genetic algorithm under the simplest reward that clears stages.

Every learned controller in this repository was shaped by an objective with many terms - progress,
velocity, an alive bonus, pit and hazard penalties, a leap bonus - and none of them concludes
`0x27` or `0x29` (10.66, 10.68). A sibling project on the Nintendo DS cleared its stages with the
opposite choice: reward *only* a new maximum of horizontal progress,

    r_t = max(0, x_t - max_{s<t} x_s) / 16,

plus a fixed bonus when the stage is cleared, and end the episode on death. Nothing rewards speed,
nothing punishes a detour, and standing still or walking back simply earns nothing - so a policy
is free to back up, wait for an enemy or a platform, and take a run-up, which is exactly what the
shaped objectives priced against.

The policy is an MLP over a compact observation read straight off the engine (velocity, ground
flag, the measured tile classes of a window ahead of Mario, and the nearest running sprites), and it
is searched by an elitist genetic algorithm, which needs no value function: the fitness of a genome
is the return of one deterministic episode from the stage's savestate. Whether a stage was cleared is
the engine's verdict, not a distance: the run leaves interactive mode `0x14` without passing through
the death mode `0x0B` and without losing a life (10.66).

Run:  python -m src.training.potential_reward_ga --state smw_map_node_68_135_t29.state
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.environment import tile_behavior, wram
from src.environment.episode_hygiene import PlayerHold
from src.utils.logging import get_logger
from src.utils.paths import CHECKPOINTS_DIR, CORE_PATH, RESULTS_DIR, ROM_PATH, state_path
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

ADDR_LIVES = 0x0DBE
ADDR_END_LEVEL_TIMER = 0x1493  # nonzero once the goal tape (or orb) is touched
GOAL_WAIT_FRAMES = 1500
GAME_MODE_DEATH = 0x0B
CLEAR_BONUS = 10.0
FRAME_SKIP = 4

# Each decision is held for FRAME_SKIP frames, buttons included. B stays down for as long as the
# policy keeps choosing a jump action, because jump height is how long B is held: releasing it
# mid-hold turns every jump into a short hop that no wide gap survives. A new jump needs one
# decision without B, which is the press-release the engine requires anyway.
ACTIONS: Tuple[Dict[str, bool], ...] = (
    {},
    {"RIGHT": True},
    {"RIGHT": True, "Y": True},
    {"RIGHT": True, "B": True},
    {"RIGHT": True, "Y": True, "B": True},
    {"LEFT": True},
    {"LEFT": True, "Y": True},
    {"B": True},
)

# Observation: 3 kinematic channels, a 7x7 tile window from one column behind Mario to five ahead,
# and the three nearest engine-running sprites (dx, dy, vx, vy each).
WINDOW_COLS = range(-1, 6)
WINDOW_ROWS = range(-3, 4)
N_SPRITES = 3
OBS_DIM = 3 + len(WINDOW_COLS) * len(WINDOW_ROWS) + 4 * N_SPRITES


def potential_reward(x: float, max_x: float) -> Tuple[float, float]:
    """The simple reward: pay only for a new maximum of x, in tiles. Returns (reward, new max)."""
    gain = max(0.0, x - max_x)
    return gain / 16.0, max(x, max_x)


@dataclass
class StageVerdict:
    """The engine's own account of how a session ended (10.66).

    A clear leaves interactive mode without the death mode on the way and with the lives counter
    untouched; a death passes through `0x0B` or takes a life. A run still in `0x14` has no verdict.
    """

    lives_at_start: int
    saw_death_mode: bool = False
    left_interactive: bool = False
    lives_now: int = -1

    def observe(self, mode: int, lives: int) -> None:
        self.lives_now = lives
        if mode == GAME_MODE_DEATH:
            self.saw_death_mode = True
        if mode != wram.GAME_MODE_INTERACTIVE:
            self.left_interactive = True

    @property
    def died(self) -> bool:
        return self.saw_death_mode or (0 <= self.lives_now < self.lives_at_start)

    @property
    def cleared(self) -> bool:
        return self.left_interactive and not self.died


def observe(emu: Any) -> np.ndarray:
    """The compact observation, read off the engine every decision."""
    s = emu.get_smw_state()
    x, y = float(s["x"]), float(s["y"])
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = s["vx"] / 48.0
    obs[1] = s["vy"] / 64.0
    obs[2] = float(s["c_ground"])
    cx, cy = int(x + 8) // wram.TILE_SIZE_PX, int(y + 16) // wram.TILE_SIZE_PX
    k = 3
    for dy in WINDOW_ROWS:
        for dx in WINDOW_COLS:
            tile = emu.read_tile_id(cx + dx, cy + dy)
            cls = tile_behavior.patch_class(tile) if tile is not None else tile_behavior.PATCH_AIR
            if cls == tile_behavior.PATCH_HAZARD:
                obs[k] = -1.0
            elif cls != tile_behavior.PATCH_AIR:
                obs[k] = 1.0
            k += 1
    sprites = sorted(emu.get_active_sprites(), key=lambda sp: abs(sp["x"] - x) + abs(sp["y"] - y))[
        :N_SPRITES
    ]
    for sp in sprites:
        obs[k : k + 4] = (
            np.clip((sp["x"] - x) / 128.0, -2, 2),
            np.clip((sp["y"] - y) / 128.0, -2, 2),
            sp["vx"] / 32.0,
            sp["vy"] / 32.0,
        )
        k += 4
    return obs


class MLPPolicy:
    """obs -> tanh hidden -> action logits, as one flat genome."""

    def __init__(self, hidden: int = 32):
        self.hidden = hidden
        self.shapes = [(OBS_DIM, hidden), (hidden,), (hidden, len(ACTIONS)), (len(ACTIONS),)]
        self.size = int(sum(np.prod(s) for s in self.shapes))

    def unpack(self, genome: np.ndarray) -> List[np.ndarray]:
        out, i = [], 0
        for shape in self.shapes:
            n = int(np.prod(shape))
            out.append(genome[i : i + n].reshape(shape))
            i += n
        return out

    def act(self, params: Sequence[np.ndarray], obs: np.ndarray) -> int:
        w1, b1, w2, b2 = params
        return int(np.argmax(np.tanh(obs @ w1 + b1) @ w2 + b2))


def run_episode(
    emu: Any,
    savestate: bytes,
    policy: MLPPolicy,
    genome: np.ndarray,
    max_frames: int = 8000,
    stall_frames: int = 600,
    record: bool = False,
) -> Dict[str, Any]:
    """One deterministic episode under the simple reward; the engine decides how it ended."""
    params = policy.unpack(genome)
    emu.load_state(savestate)
    verdict = StageVerdict(lives_at_start=emu.read_wram_u8(ADDR_LIVES))
    hold = PlayerHold()
    s = emu.get_smw_state()
    start_x = max_x = float(s["x"])
    ret, frames, last_gain, actions = 0.0, 0, 0, []
    while frames < max_frames:
        a = policy.act(params, observe(emu))
        if record:
            actions.append(a)
        for _ in range(FRAME_SKIP):
            emu.set_input(hold.buttons(dict(ACTIONS[a])))
            emu.step_frame()
            frames += 1
            s = emu.get_smw_state()
            hold.observe(np.array([s["x"], s["y"], s["vx"], s["vy"]]))
            verdict.observe(emu.get_game_mode(), emu.read_wram_u8(ADDR_LIVES))
            if verdict.left_interactive or verdict.died:
                break
            r, max_x = potential_reward(float(s["x"]), max_x)
            if r > 0:
                ret += r
                last_gain = frames
        if verdict.left_interactive or verdict.died:
            break
        if emu.read_wram_u8(ADDR_END_LEVEL_TIMER):
            break
        if frames - last_gain > stall_frames or hold.exhausted():
            break
    # Past the goal the engine walks Mario off and counts the score while still in 0x14; release
    # the controls and give it the frames to deliver its verdict instead of calling it a stall.
    if emu.read_wram_u8(ADDR_END_LEVEL_TIMER) and not verdict.left_interactive:
        for _ in range(GOAL_WAIT_FRAMES):
            emu.set_input({})
            emu.step_frame()
            verdict.observe(emu.get_game_mode(), emu.read_wram_u8(ADDR_LIVES))
            if verdict.left_interactive:
                break
    if verdict.cleared:
        ret += CLEAR_BONUS
    out: Dict[str, Any] = {
        "return": float(ret),
        "progress_px": float(max_x - start_x),
        "max_x": float(max_x),
        "frames": int(frames),
        "cleared": bool(verdict.cleared),
        "died": bool(verdict.died),
    }
    if record:
        out["actions"] = actions
    return out


# --- worker pool: one emulator per process, because the core's state is process-global ---------

_WORKER: Dict[str, Any] = {}


def _init_worker(state_bytes: bytes, hidden: int, max_frames: int) -> None:
    from src.environment.snes_emulator import SnesLibretroEmulator

    emu = SnesLibretroEmulator(CORE_PATH)
    emu.load_rom(ROM_PATH)
    _WORKER.update(emu=emu, state=state_bytes, policy=MLPPolicy(hidden), max_frames=max_frames)


def _evaluate(genome: np.ndarray) -> Dict[str, Any]:
    w = _WORKER
    return run_episode(w["emu"], w["state"], w["policy"], genome, max_frames=w["max_frames"])


def train_potential_reward_ga(
    state_name: str,
    generations: int = 150,
    population: int = 48,
    elite: int = 6,
    sigma: float = 0.15,
    hidden: int = 32,
    max_frames: int = 8000,
    workers: int = 12,
    seed: int = 0,
    output_dir: str = RESULTS_DIR,
    checkpoint_dir: str = CHECKPOINTS_DIR,
    stop_on_clear: bool = True,
) -> Dict[str, Any]:
    """Elitist GA over the MLP genome with fitness = simple-reward return of one episode."""
    rng = np.random.default_rng(seed)
    with open(state_path(state_name), "rb") as handle:
        state_bytes = handle.read()
    policy = MLPPolicy(hidden)
    pop = [rng.normal(0.0, 0.5, policy.size).astype(np.float32) for _ in range(population)]
    history: List[Dict[str, Any]] = []
    best: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None
    first_clear: Optional[int] = None
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_init_worker, initargs=(state_bytes, hidden, max_frames)
    ) as pool:
        for gen in range(generations):
            results = pool.map(_evaluate, pop)
            order = np.argsort([-r["return"] for r in results])
            top = results[order[0]]
            if best is None or top["return"] > best[1]["return"]:
                best = (pop[order[0]].copy(), top)
            if top["cleared"] and first_clear is None:
                first_clear = gen
            row = {
                "generation": gen,
                "best_return": top["return"],
                "best_progress_px": top["progress_px"],
                "mean_return": float(np.mean([r["return"] for r in results])),
                "clears": int(sum(r["cleared"] for r in results)),
                "elapsed_s": round(time.time() - t0, 1),
            }
            history.append(row)
            logger.info(
                "gen %3d | best %.1f px (ret %.2f) | mean ret %.2f | clears %d/%d | %.0fs",
                gen,
                top["progress_px"],
                top["return"],
                row["mean_return"],
                row["clears"],
                population,
                row["elapsed_s"],
            )
            if stop_on_clear and first_clear is not None:
                break
            elites = [pop[i] for i in order[:elite]]
            children = [e.copy() for e in elites]
            while len(children) < population:
                parent = elites[int(rng.integers(0, elite))]
                children.append(parent + rng.normal(0.0, sigma, policy.size).astype(np.float32))
            pop = children
    assert best is not None
    genome, best_result = best
    tag = os.path.splitext(os.path.basename(state_name))[0]
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = os.path.join(checkpoint_dir, f"potential_ga_{tag}.npz")
    np.savez(ckpt, genome=genome, hidden=hidden)
    payload = {
        "study": "Genetic-algorithm policy under the simple potential-maximum reward",
        "protocol": {
            "reward": "max(0, x_t - max_x_so_far) / 16, + 10 on an engine-verified clear",
            "termination": "death (mode 0x0B or a life lost), leaving 0x14, 600-frame stall",
            "state": state_name,
            "generations_run": len(history),
            "population": population,
            "elite": elite,
            "sigma": sigma,
            "hidden": hidden,
            "frame_skip": FRAME_SKIP,
            "obs_dim": OBS_DIM,
            "n_actions": len(ACTIONS),
            "parameters": policy.size,
            "max_frames": max_frames,
            "seed": seed,
        },
        "first_clear_generation": first_clear,
        "best": best_result,
        "history": history,
        "checkpoint": os.path.relpath(ckpt, os.path.dirname(output_dir) or ".").replace(
            os.sep, "/"
        ),
    }
    write_metrics(
        os.path.join(output_dir, f"potential_ga_{tag}_metrics.json"),
        payload,
        seed=seed,
        command=f"python -m src.training.potential_reward_ga --state {state_name}",
    )
    return payload


if __name__ == "__main__":
    from src.utils.config import parse_args_with_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--config", default=None)
    parser.add_argument("--state", default="smw_map_node_68_135_t29.state")
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINTS_DIR)
    parser.add_argument("--keep-going", action="store_true", help="do not stop at the first clear")
    args = parse_args_with_config(parser)
    train_potential_reward_ga(
        args.state,
        generations=args.generations,
        population=args.population,
        elite=args.elite,
        sigma=args.sigma,
        hidden=args.hidden,
        max_frames=args.max_frames,
        workers=args.workers,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        stop_on_clear=not args.keep_going,
    )
