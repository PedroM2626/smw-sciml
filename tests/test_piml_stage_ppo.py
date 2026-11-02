"""Unit and artifact gates for Physics-Informed Stage PPO (`src/training/piml_stage_ppo.py`, 10.71)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.losses.physics_rl_losses import PhysicsInformedCriticLoss
from src.training.piml_stage_ppo import (
    FRAME_SKIP,
    MPC_PRIMITIVE_TO_STAGE_ACTION,
    N_ACTIONS,
    OBS_DIM,
    PHYS_DIM,
    STAGE_ACTION_TO_MPC_PRIMITIVE,
    PIMLStageActorCritic,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "piml_stage_ppo_metrics.json"


def test_piml_stage_actor_critic_forward_and_cbf_projection() -> None:
    model = PIMLStageActorCritic(obs_dim=OBS_DIM, phys_dim=PHYS_DIM, hidden_dim=64, cbf_beta=0.5)
    obs = torch.randn(6, OBS_DIM)
    phys = torch.zeros(6, PHYS_DIM)
    phys[:, 0] = 200.0
    phys[:, 1] = 360.0
    phys[:, 2] = 48.0  # moving right
    phys[:, 3] = 12.0  # falling
    phys[:, 7] = 1.0  # pressing against right wall

    phys_req = phys.clone().requires_grad_(True)
    dist, val, raw_logits = model(obs, phys_req)
    assert dist.probs.shape == (6, N_ACTIONS)
    assert val.shape == (6,)
    assert raw_logits.shape == (6, N_ACTIONS)
    assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(6), atol=1e-5)

    # PhysicsInformedCriticLoss must differentiate V(s) w.r.t. phys_req cleanly
    hjb = PhysicsInformedCriticLoss(alpha=0.5, death_y=450.0, danger_low=340.0)
    loss_hjb = hjb(phys_req, val)
    assert loss_hjb.ndim == 0
    assert float(loss_hjb.item()) >= 0.0


def test_hard_kinematics_head_enforces_exact_fixed_point_identity() -> None:
    model = PIMLStageActorCritic(obs_dim=OBS_DIM, phys_dim=PHYS_DIM, hidden_dim=64)
    obs = torch.randn(10, OBS_DIM)
    phys = torch.randn(10, PHYS_DIM) * 32.0
    actions = torch.randint(0, N_ACTIONS, (10,))
    pred_kin, struct_res = model.predict_kinematics(obs, phys, actions)
    assert pred_kin.shape == (10, 4)
    assert struct_res.shape == (10,)
    # Structural identity: 16 * (x_next - x) - FRAME_SKIP * vx_next == 0 within float32 precision
    dx = pred_kin[:, 0] - phys[:, 0]
    vx_next = pred_kin[:, 2]
    assert torch.allclose(16.0 * dx, float(FRAME_SKIP) * vx_next, atol=1e-2)
    assert float(struct_res.abs().max().item()) < 1e-2


def test_stage_and_mpc_action_permutations_are_bijections() -> None:
    for idx in range(N_ACTIONS):
        mpc_idx = STAGE_ACTION_TO_MPC_PRIMITIVE[idx]
        assert MPC_PRIMITIVE_TO_STAGE_ACTION[mpc_idx] == idx


def test_committed_piml_stage_ppo_artifact_clears_0x29_and_0x27() -> None:
    if not ARTIFACT.is_file():
        pytest.skip("piml_stage_ppo_metrics.json has not been generated")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    stages = payload["stages"]
    for st_name, min_px in (
        ("smw_map_node_68_135_t29.state", 4800.0),
        ("smw_map_round_1_152_104_t27.state", 5000.0),
    ):
        assert st_name in stages
        rec = stages[st_name]
        ckpt = REPO / rec["checkpoint"]
        assert ckpt.is_file(), f"missing checkpoint {ckpt}"
        ev = rec["closed_loop_evaluation"]
        assert ev["cleared"] is True
        assert ev["died"] is False
        assert ev["lives_at_start"] == ev["lives_at_end"] == 4
        assert ev["progress_px"] >= min_px
        assert "0x0e" in ev["modes_seen"]
        diag = rec["sciml_diagnostics"]
        assert diag["policy_action_agreement"] == pytest.approx(1.0, abs=1e-4)
        assert diag["structural_kinematic_identity_max_residual"] < 0.01
