"""Emulator-free tests for the simple-reward GA (src/training/potential_reward_ga.py).

The reward, the engine verdict and the policy are what a clear claimed by this module rests on, so
each is pinned here without a ROM: the reward pays only for new progress, the verdict reads the
engine's bytes the way 10.66 defines a clear, and an episode run against a scripted fake emulator
ends for the reason the bytes give.
"""

import os

import numpy as np
import pytest

from src.environment import wram
from src.training import potential_reward_ga as ga


def test_reward_pays_only_for_a_new_maximum() -> None:
    r, m = ga.potential_reward(32.0, 0.0)
    assert (r, m) == (2.0, 32.0)
    r, m = ga.potential_reward(10.0, m)  # walking back earns nothing and loses nothing
    assert (r, m) == (0.0, 32.0)
    r, m = ga.potential_reward(48.0, m)
    assert (r, m) == (1.0, 48.0)


def test_verdict_separates_a_clear_from_a_death_and_from_no_verdict() -> None:
    running = ga.StageVerdict(lives_at_start=4)
    running.observe(wram.GAME_MODE_INTERACTIVE, 4)
    assert not running.cleared and not running.died

    cleared = ga.StageVerdict(lives_at_start=4)
    for mode in (0x14, 0x0C, 0x0D, 0x0E):
        cleared.observe(mode, 4)
    assert cleared.cleared and not cleared.died

    dead = ga.StageVerdict(lives_at_start=4)
    for mode in (0x14, ga.GAME_MODE_DEATH, 0x0C):
        dead.observe(mode, 4)
    assert dead.died and not dead.cleared

    lost_a_life = ga.StageVerdict(lives_at_start=4)
    lost_a_life.observe(0x0C, 3)
    assert lost_a_life.died and not lost_a_life.cleared


def test_policy_round_trips_its_genome_and_picks_a_valid_action() -> None:
    policy = ga.MLPPolicy(hidden=8)
    genome = np.arange(policy.size, dtype=np.float32) / policy.size
    params = policy.unpack(genome)
    assert [p.shape for p in params] == [tuple(s) for s in policy.shapes]
    assert np.concatenate([p.ravel() for p in params]).tolist() == genome.tolist()
    action = policy.act(params, np.ones(ga.OBS_DIM, dtype=np.float32))
    assert 0 <= action < len(ga.ACTIONS)


class _FakeEmu:
    """Walks right one pixel per frame, then plays a scripted mode sequence."""

    def __init__(self, modes, lives_after=4, x_cap=1e9):
        self.modes = list(modes)
        self.lives_after = lives_after
        self.x_cap = x_cap
        self.t = 0
        self.inputs = []

    def load_state(self, blob):
        self.t = 0

    def read_wram_u8(self, addr):
        if addr == ga.ADDR_LIVES:
            return 4 if self.t < len(self.modes) else self.lives_after
        return 0  # no goal tape

    def get_smw_state(self):
        x = min(16.0 + self.t, self.x_cap)
        return {"x": x, "y": 300.0, "vx": 16.0, "vy": 0.0, "c_ground": 1.0}

    def read_tile_id(self, tx, ty):
        return None

    def get_active_sprites(self):
        return [{"x": 40.0, "y": 300.0, "vx": -8.0, "vy": 0.0}]

    def set_input(self, buttons):
        self.inputs.append(buttons)

    def step_frame(self):
        self.t += 1

    def get_game_mode(self):
        return self.modes[self.t] if self.t < len(self.modes) else 0x0E


def test_episode_reward_is_progress_over_sixteen_and_a_clear_adds_the_bonus() -> None:
    emu = _FakeEmu([wram.GAME_MODE_INTERACTIVE] * 160 + [0x0C])
    out = ga.run_episode(emu, b"", ga.MLPPolicy(4), np.zeros(ga.MLPPolicy(4).size, np.float32))
    assert out["cleared"] and not out["died"]
    assert out["progress_px"] == pytest.approx(159.0)
    assert out["return"] == pytest.approx(159.0 / 16.0 + ga.CLEAR_BONUS)


def test_episode_ends_on_death_without_the_bonus() -> None:
    emu = _FakeEmu([wram.GAME_MODE_INTERACTIVE] * 80 + [ga.GAME_MODE_DEATH], lives_after=3)
    out = ga.run_episode(emu, b"", ga.MLPPolicy(4), np.zeros(ga.MLPPolicy(4).size, np.float32))
    assert out["died"] and not out["cleared"]
    assert out["return"] == pytest.approx(out["progress_px"] / 16.0)


def test_episode_ends_on_a_stall() -> None:
    emu = _FakeEmu([wram.GAME_MODE_INTERACTIVE] * 5000, x_cap=100.0)
    out = ga.run_episode(
        emu, b"", ga.MLPPolicy(4), np.zeros(ga.MLPPolicy(4).size, np.float32), stall_frames=200
    )
    assert not out["cleared"] and not out["died"]
    assert out["frames"] < 400


def test_observation_has_the_declared_width_and_sees_the_sprite() -> None:
    obs = ga.observe(_FakeEmu([wram.GAME_MODE_INTERACTIVE]))
    assert obs.shape == (ga.OBS_DIM,)
    sprite = obs[ga.OBS_DIM - 4 * ga.N_SPRITES :][:4]
    assert sprite[0] == pytest.approx(24.0 / 128.0) and sprite[2] == pytest.approx(-0.25)


ARTIFACTS = {
    "0x29": "potential_ga_smw_map_node_68_135_t29_metrics.json",
    "0x27": "potential_ga_smw_map_round_1_152_104_t27_metrics.json",
}


@pytest.mark.parametrize("stage", sorted(ARTIFACTS))
def test_section_10_69_quotes_the_committed_artifacts(stage: str) -> None:
    """Every number in 10.69's table is the artifact's, and the verdict column is the artifact's."""
    import json

    from src.utils.document import full_document
    from src.utils.paths import RESULTS_DIR

    path = os.path.join(RESULTS_DIR, ARTIFACTS[stage])
    if not os.path.isfile(path):
        pytest.skip("the simple-reward GA has not been run")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    text = full_document()
    best = payload["best"]
    row = next(line for line in text.split("\n") if line.startswith(f"| `{stage}` |"))
    assert f"{best['progress_px']:.4f} px" in row
    assert f"{best['frames']} frames" in row
    assert f"{payload['protocol']['parameters']} weights" in text
    if best["cleared"]:
        assert f"cleared in generation {payload['first_clear_generation']}" in row
        assert "**cleared**" in row
    else:
        assert payload["first_clear_generation"] is None
        assert f"{payload['protocol']['generations_run']}, no clear" in row
