"""Tests for TD-MPC terminal value, reflex rules, and diagnose helpers."""

import numpy as np
import torch

from src.environment.episode_hygiene import PlayerHold
from src.evaluation.diagnose_obstacle_1000 import min_takeoff_vx
from src.evaluation.mpc_reflex_ablation import apply_reflexes
from src.planning.terminal_value import (
    TerminalValueNet,
    TerminalValueObjective,
    compute_mc_returns,
    fit_terminal_value,
)


def _base_state(**over):
    s = {
        "delta_x_enemy": 999.0,
        "hazard_active": 0.0,
        "c_ground": 1.0,
        "c_right": 0.0,
    }
    s.update(over)
    return s


def test_mc_returns_hand_computed():
    progress = np.array([0.0, 1.0, 3.0, 6.0])
    rets = compute_mc_returns(progress, gamma=1.0)
    assert rets.tolist() == [6.0, 6.0, 5.0, 3.0]
    rets_d = compute_mc_returns(progress, gamma=0.5)
    assert rets_d.tolist() == [1.375, 2.75, 3.5, 3.0]


def test_fit_terminal_value_learns():
    rng = np.random.default_rng(0)
    states = rng.uniform(-5, 5, size=(64, 4)).astype(np.float32)
    returns = (2.0 * states[:, 0] - states[:, 1]).astype(np.float64)
    net, stats = fit_terminal_value(states, returns, hidden_dim=32, epochs=150)
    with torch.no_grad():
        pred = net(torch.tensor(states)).numpy() * stats["target_std"] + stats["target_mean"]
    ss_res = np.sum((returns - pred) ** 2)
    ss_tot = np.sum((returns - returns.mean()) ** 2)
    assert 1.0 - ss_res / ss_tot > 0.9


def test_terminal_objective_prefers_high_value_end():
    torch.manual_seed(0)
    net = TerminalValueNet(hidden_dim=16)
    # Isolate the terminal term: no progress/velocity/waypoint shaping.
    obj = TerminalValueObjective(
        value_net=net,
        terminal_weight=5.0,
        gamma=1.0,
        weight_progress=0.0,
        weight_velocity=0.0,
        weight_target=0.0,
        arrival_bonus=0.0,
    )
    init = torch.zeros(2, 8)
    # Small-scale inputs: y=300 would saturate the random Tanh net.
    candidates = torch.tensor([[5.0, 3.0, 3.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    with torch.no_grad():
        values = net(candidates)
    assert not torch.allclose(values[0], values[1])  # non-degenerate random net
    hi = int(torch.argmax(values).item())
    lo = 1 - hi
    trajs = torch.zeros(2, 5, 8)
    trajs[hi, :, :4] = candidates[hi]
    trajs[lo, :, :4] = candidates[lo]
    rewards = obj.compute_trajectory_rewards(init, trajs)
    assert rewards[hi] > rewards[lo]


def test_reflex_wall_vault_fires():
    act, prev, hits = apply_reflexes({"RIGHT": True}, _base_state(c_right=1.0), 3, False)
    assert act["B"] and act["Y"] and act["RIGHT"]
    assert hits["wall_vault"] == 1 and hits["hazard_vault"] == 0


def test_reflex_hazard_vault_fires():
    act, _, hits = apply_reflexes(
        {"RIGHT": True}, _base_state(hazard_active=1.0, delta_x_enemy=40.0), 3, False
    )
    assert act["B"] and hits["hazard_vault"] == 1


def test_reflex_b_pulse_on_even_frames_only():
    act_even, _, hits_even = apply_reflexes({"B": True}, _base_state(), 4, True)
    act_odd, _, hits_odd = apply_reflexes({"B": True}, _base_state(), 5, True)
    assert act_even["B"] is False and hits_even["b_pulse"] == 1
    assert act_odd["B"] is True and hits_odd["b_pulse"] == 0


def test_reflex_pure_passthrough():
    act, prev, hits = apply_reflexes({"RIGHT": True}, _base_state(), 3, False)
    assert act == {"RIGHT": True} and prev is False
    assert sum(hits.values()) == 0


def test_min_takeoff_vx_physics():
    assert min_takeoff_vx(0.0) == 0.0
    assert min_takeoff_vx(40.0) < min_takeoff_vx(80.0)
    # The 83px diagnosis gap needs running speed: above walk (20), below run (48).
    v = min_takeoff_vx(82.75)
    assert 20.0 < v < 48.0


# ---------- the opt-in message-box operator (10.60) ----------


def _observed(hold: PlayerHold, frames: int, vector):
    for _ in range(frames):
        hold.observe(vector)
    return hold


def test_reflex_without_operator_keeps_the_published_contract():
    """No PlayerHold passed: exactly the published three reflexes, no A channel."""
    act, _, hits = apply_reflexes({"RIGHT": True}, _base_state(), 3, False)
    assert "A" not in act
    assert set(hits) == {"hazard_vault", "wall_vault", "b_pulse"}


def test_reflex_message_box_pulses_a_once_the_hold_is_long_enough():
    hold = _observed(PlayerHold(trigger=3, limit=8), 4, np.zeros(8))
    act, _, hits = apply_reflexes({"RIGHT": True}, _base_state(), 0, False, message_box=hold)
    assert act["A"] is True
    assert hits["message_box"] == 1


def test_reflex_message_box_is_quiet_below_the_trigger():
    """One identical record is not a hold: the key exists, the pulse does not."""
    hold = PlayerHold(trigger=5)
    hold.observe(np.zeros(8))
    act, _, hits = apply_reflexes({"RIGHT": True}, _base_state(), 0, False, message_box=hold)
    assert "A" not in act
    assert hits["message_box"] == 0


def test_reflex_message_box_follows_the_shared_duty_cycle():
    """The A pulse is PlayerHold's decision, not a reflex-layer rule of its own.

    The expected pattern is derived from the published duty cycle (on for 4 of every
    8 frames of a hold, never before the trigger) so this test fails if the reflex
    layer adds, drops or shifts a frame of the operator's pulse. The first observe()
    only installs the reference record, so the run length after the k-th call is
    k - 1 - the same convention the recorders are gated on.
    """
    hold = PlayerHold(trigger=2, limit=32)
    vector = np.zeros(8)
    pressed = []
    for _ in range(16):
        hold.observe(vector)
        act, _, _ = apply_reflexes({}, _base_state(), 0, False, message_box=hold)
        pressed.append(bool(act.get("A", False)))
    expected = [run >= 2 and (run % 8) < 4 for run in range(0, 16)]
    assert pressed == expected


def test_reflex_message_box_gives_up_after_the_limit():
    hold = PlayerHold(trigger=1, limit=2)
    pressed = []
    for _ in range(4):
        hold.observe(np.zeros(8))
        act, _, _ = apply_reflexes({}, _base_state(), 0, False, message_box=hold)
        pressed.append(bool(act.get("A", False)))
    assert pressed == [False, True, True, False]
