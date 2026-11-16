"""Emulator-free unit tests for src/training/piml_stage_ppo.py and the 10.75 ablation.

The module was covered only by a test that reads its committed artifact. These drive its pure
pieces and its closed-loop evaluation against a scripted fake console, which is where the verdict
of 10.74.5 lives: a pipe into a sub-level must neither end nor clear the episode.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from src.evaluation import piml_sciml_ablation as abl
from src.training import piml_stage_ppo as pp


class _FakeConsole:
    """A scripted console: x advances every frame, the mode byte follows a script by frame."""

    def __init__(self, modes: Dict[int, int], timer_at: int | None = None, lives: int = 4):
        self.modes = modes
        self.timer_at = timer_at
        self.lives = lives
        self.t = 0
        self.mode = 0x14
        self.inputs: List[Dict[str, bool]] = []

    def load_state(self, blob: bytes) -> None:
        self.t, self.mode = 0, 0x14

    def read_wram_u8(self, addr: int) -> int:
        if addr == pp.ADDR_LIVES:
            return self.lives
        if addr == pp.ADDR_END_LEVEL_TIMER:
            return 255 if self.timer_at is not None and self.t >= self.timer_at else 0
        return 0

    def get_smw_state(self) -> Dict[str, float]:
        return {
            "x": 16.0 + self.t * 0.5,
            "y": 300.0,
            "vx": 8.0,
            "vy": 0.0,
            "c_ground": 1.0,
            "c_ceiling": 0.0,
            "c_left": 0.0,
            "c_right": 0.0,
        }

    def read_tile_id(self, tx: int, ty: int) -> int | None:
        return 0x0100 if ty > 20 else None

    def get_active_sprites(self) -> List[Dict[str, float]]:
        return [{"x": 60.0, "y": 300.0, "vx": -4.0, "vy": 0.0}]

    def set_input(self, buttons: Dict[str, bool]) -> None:
        self.inputs.append(buttons)

    def step_frame(self) -> None:
        self.t += 1
        self.mode = self.modes.get(self.t, self.mode)

    def get_game_mode(self) -> int:
        return self.mode


def _model() -> pp.PIMLStageActorCritic:
    torch.manual_seed(0)
    return pp.PIMLStageActorCritic(hidden_dim=16)


def test_macro_buttons_are_held_on_land_and_pulsed_in_water() -> None:
    land = [pp.get_macro_subframe_buttons(4, sub) for sub in range(pp.FRAME_SKIP)]
    assert all(b == land[0] for b in land)
    water = [pp.get_macro_subframe_buttons(4, sub, is_aquatic=True) for sub in range(pp.FRAME_SKIP)]
    assert [bool(b.get("B")) for b in water] == [True, True, False, False]
    rising = [pp.get_macro_subframe_buttons(7, sub, is_aquatic=True) for sub in range(4)]
    assert all(b["UP"] for b in rising) and rising[0]["B"] and not rising[3].get("B")
    assert pp.get_macro_subframe_buttons(2, 3, is_aquatic=True) == {"RIGHT": True, "Y": True}
    jump = pp.get_macro_subframe_buttons(3, 3, is_aquatic=True)  # B released on the late subframes
    assert "B" not in jump or not jump["B"]


def test_gae_matches_a_hand_computation() -> None:
    adv, ret = pp.compute_gae(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.25], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        next_value=9.0,
        gamma=0.5,
        gae_lambda=1.0,
    )
    # last step terminates: delta1 = 0 - 0.25; first: 1 + 0.5 * 0.25 - 0.5 = 0.625, plus 0.5 * delta1
    assert adv[1] == pytest.approx(-0.25)
    assert adv[0] == pytest.approx(0.625 + 0.5 * -0.25)
    assert ret == pytest.approx(adv + np.array([0.5, 0.25]))


def test_the_hold_trigger_is_per_stage() -> None:
    assert pp.hold_trigger_for("smw_map_round_2_184_72_t26.state") == 60
    assert pp.hold_trigger_for("smw_map_node_68_135_t29.state") == pp.DEFAULT_HOLD_TRIGGER


def test_the_observation_has_the_declared_widths_and_reads_the_tracker() -> None:
    tracker = pp.EpisodeTracker(
        max_x=40.0, stall_steps=10, b_hold_steps=8, prev_action=3, step_count=80
    )
    obs, phys = pp.extract_piml_observation(_FakeConsole({}), tracker)
    assert obs.shape == (pp.OBS_DIM,) and phys.shape == (pp.PHYS_DIM,)
    assert obs[12 + 3] == 1.0 and obs[9] == pytest.approx(10 / 50.0)
    assert obs[11] == pytest.approx(80 / 800.0)
    clone = tracker.copy()
    clone.stall_steps = 99
    assert tracker.stall_steps == 10


def test_the_actor_critic_returns_a_distribution_a_value_and_a_consistent_kinematic_head() -> None:
    model = _model()
    obs = torch.zeros(5, pp.OBS_DIM)
    phys = torch.tensor([[100.0, 300.0, 16.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).repeat(5, 1)
    dist, value, _ = model(obs, phys)
    assert dist.probs.shape == (5, pp.N_ACTIONS) and torch.allclose(
        dist.probs.sum(-1), torch.ones(5)
    )
    assert value.shape == (5,)
    actions = torch.arange(5) % pp.N_ACTIONS
    pred, residual = model.predict_kinematics(obs, phys, actions)
    assert pred.shape[0] == 5 and torch.isfinite(pred).all()
    assert residual.abs().max() < 1e-3  # the integration identity holds by construction


def test_a_macro_step_advances_four_frames_and_pays_for_progress() -> None:
    console = _FakeConsole({})
    verdict = pp.StageVerdict(lives_at_start=4)
    hold = pp.PlayerHold()
    tracker = pp.EpisodeTracker.from_emu(console)
    reward, done = pp.step_macro_action(console, hold, verdict, tracker, 1)
    assert console.t == pp.FRAME_SKIP and len(console.inputs) == pp.FRAME_SKIP
    assert reward > 0 and not done and tracker.step_count == 1


def test_the_evaluation_walks_through_a_pipe_and_clears_only_at_the_level_end() -> None:
    """10.74.5: modes 0x0F-0x13 are a sub-level transition, not the goal."""
    pipe = {40: 0x0F, 44: 0x10, 45: 0x11, 48: 0x12, 52: 0x13, 60: 0x14}
    finish = {**pipe, 200: 0x0C, 230: 0x0D, 250: 0x0E}
    stuck_in_pipe = _FakeConsole(pipe)  # reaches the pipe and never the goal
    out = pp.evaluate_closed_loop_piml(
        stuck_in_pipe, b"", _model(), torch.device("cpu"), max_frames=160, post_verdict_frames=0
    )
    assert not out["cleared"] and not out["died"]
    assert out["modes_seen"][:6] == ["0x14", "0x0f", "0x10", "0x11", "0x12", "0x13"]

    cleared = _FakeConsole(finish, timer_at=120)
    out = pp.evaluate_closed_loop_piml(
        cleared, b"", _model(), torch.device("cpu"), max_frames=400, post_verdict_frames=0
    )
    assert out["cleared"] and not out["died"]
    assert out["end_timer_frame"] is not None and "0x0c" in out["modes_seen"]
    assert out["lives_at_start"] == out["lives_at_end"] == 4


def test_a_death_is_not_a_clear() -> None:
    dying = _FakeConsole({40: 0x0B, 70: 0x0C}, lives=3)
    dying_lives = [4, 3]

    def lives(addr: int, _orig: Any = dying.read_wram_u8) -> int:
        return dying_lives[0 if dying.t < 40 else 1] if addr == pp.ADDR_LIVES else _orig(addr)

    dying.read_wram_u8 = lives  # type: ignore[method-assign]
    out = pp.evaluate_closed_loop_piml(
        dying, b"", _model(), torch.device("cpu"), max_frames=200, post_verdict_frames=0
    )
    assert out["died"] and not out["cleared"]


# --- the 10.75 ablation: its arms and its table are pure -----------------------------------------


def test_the_arms_switch_off_exactly_what_they_say() -> None:
    assert abl.ARMS["full"] == {}
    sciml = {"cbf_beta", "lambda_hjb", "lambda_phys_act", "lambda_pinn"}
    assert set(abl.ARMS["no_sciml"]) == sciml and not any(abl.ARMS["no_sciml"].values())
    assert abl.ARMS["no_imitation"] == {"lambda_sil": 0.0, "polish": False}
    assert set(abl.ARMS["no_imitation_no_sciml"]) == sciml | {"lambda_sil", "polish"}


def _run(stage: str, arm: str, seed: int, ppo_ok: bool, final_ok: bool) -> Dict[str, Any]:
    def cell(ok: bool) -> Dict[str, Any]:
        return {"cleared": ok, "died": not ok, "progress_px": 5000.0 if ok else 400.0, "frames": 10}

    return {
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "after_ppo": cell(ppo_ok),
        "final": cell(final_ok),
        "polish_steps": 100,
        "anchor_accuracy_after_ppo": 0.99,
        "policy_action_agreement": 1.0,
    }


def test_the_summary_counts_clears_after_ppo_and_after_the_polish() -> None:
    stage = "smw_map_node_68_135_t29.state"
    runs = [_run(stage, "full", s, ppo_ok=(s == 1), final_ok=True) for s in range(3)]
    runs += [_run(stage, "no_imitation", s, ppo_ok=False, final_ok=False) for s in range(3)]
    summary = abl.summarise(runs)[stage]
    assert summary["full"]["cleared_after_ppo"] == 1 and summary["full"]["cleared_final"] == 3
    assert summary["no_imitation"]["cleared_final"] == 0
    rows = abl.render_table({stage: summary})
    assert rows[0].startswith("| `0x29` | `full` | 1 / 3 |") and rows[0].endswith("| 100 |")


class _TrainableConsole(_FakeConsole):
    """The console the trainer also needs: a ROM to load, a savestate to take, a core to close."""

    def __init__(self, core: str | None = None):
        super().__init__({})

    def load_rom(self, path: str | None = None) -> None:
        return None

    def save_state(self) -> bytes:
        return b"snapshot"

    def close(self) -> None:
        return None


def _patched_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> str:
    state = tmp_path / "smw_map_round_1_152_104_t27.state"
    state.write_bytes(b"state")
    monkeypatch.setattr(pp, "state_path", lambda name: str(state))
    monkeypatch.setattr(pp, "extract_reference_actions", lambda emu, root, name: [1, 2, 4, 3] * 12)
    monkeypatch.setattr("src.environment.snes_emulator.SnesLibretroEmulator", _TrainableConsole)
    return state.name


def test_a_short_training_run_exercises_every_stage_of_the_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """PPO epochs, the anchor terms, the polish and the closed-loop evaluation, on a fake console."""
    name = _patched_trainer(monkeypatch, tmp_path)
    result = pp.train_piml_stage_ppo_single(
        name,
        epochs=2,
        rollout_steps_per_epoch=8,
        ppo_updates_per_epoch=2,
        hidden_dim=16,
        device_str="cpu",
        checkpoint_dir=str(tmp_path),
        require_clear=False,
        pre_polish_eval=True,
    )
    diag = result["training_diagnostics"]
    assert diag["polish"] and diag["polish_steps"] > 0
    assert diag["pre_polish_evaluation"] is not None
    assert result["closed_loop_evaluation"]["cleared"] is False  # the fake console has no goal
    assert result["sciml_diagnostics"]["policy_action_agreement"] > 0.5
    blob = torch.load(tmp_path / "piml_ppo_smw_map_round_1_152_104_t27.pt", weights_only=False)
    assert blob["hold_trigger"] == pp.DEFAULT_HOLD_TRIGGER and "closed_loop_actions" in blob


def test_the_ablation_switches_skip_the_polish_and_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    name = _patched_trainer(monkeypatch, tmp_path)
    result = pp.train_piml_stage_ppo_single(
        name,
        epochs=1,
        rollout_steps_per_epoch=8,
        ppo_updates_per_epoch=1,
        hidden_dim=16,
        device_str="cpu",
        checkpoint_dir=str(tmp_path),
        require_clear=False,
        save_checkpoint=False,
        **abl.ARMS["no_imitation_no_sciml"],
    )
    assert result["training_diagnostics"]["polish_steps"] == 0
    assert (
        result["sciml_diagnostics"]["cbf_beta"] == 0.0
        and result["training_diagnostics"]["lambda_sil"] == 0.0
    )
    assert not list(tmp_path.glob("*.pt"))


def test_a_required_clear_that_does_not_happen_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    name = _patched_trainer(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="failed to clear"):
        pp.train_piml_stage_ppo_single(
            name,
            epochs=1,
            rollout_steps_per_epoch=8,
            ppo_updates_per_epoch=1,
            hidden_dim=16,
            device_str="cpu",
            checkpoint_dir=str(tmp_path),
            save_checkpoint=False,
        )


def test_section_10_75_is_the_ablation_artifact() -> None:
    """Every row of 10.75's table and each claim it draws is derived from the committed artifact."""
    import json
    import os

    from src.utils.document import full_document
    from src.utils.paths import RESULTS_DIR

    path = os.path.join(RESULTS_DIR, "piml_sciml_ablation_metrics.json")
    if not os.path.isfile(path):
        pytest.skip("the ablation has not been run")
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    text = full_document()
    missing = [row for row in abl.render_table(artifact["summary"]) if row not in text]
    assert not missing, "10.75's table disagrees with the artifact:\n" + "\n".join(missing)

    runs = artifact["runs"]

    def cleared(arm: str, key: str) -> int:
        return sum(bool(r[key]["cleared"]) for r in runs if r["arm"] == arm)

    imitation = ("full", "no_sciml")
    assert sum(cleared(a, "final") for a in ("no_imitation", "no_imitation_no_sciml")) == 0
    assert "0 of 9 for each arm" in text and "The 18 `no_imitation`" in text
    assert sum(cleared(a, "after_ppo") for a in imitation) == 2 and "2 clear right after" in text
    assert sum(cleared(a, "final") for a in imitation) == 12 and "12 after the polish" in text
    assert (cleared("full", "final"), cleared("no_sciml", "final")) == (7, 5)
    assert "`full` clears 7 of 9 runs and `no_sciml` 5 of 9" in text
    furthest = max(r["final"]["progress_px"] for r in runs if r["arm"].startswith("no_imitation"))
    assert f"{furthest:.0f} px" in text
    accuracies = [r["anchor_accuracy_after_ppo"] for r in runs if r["arm"] in imitation]
    steps = [r["polish_steps"] for r in runs if r["arm"] in imitation]
    assert f"{min(accuracies):.0%} to {max(accuracies):.1%}" in text
    assert f"between {min(steps)} and {max(steps)} cross-entropy steps" in text
