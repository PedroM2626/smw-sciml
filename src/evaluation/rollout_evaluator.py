"""
rollout_evaluator.py
Rigorous long-horizon trajectory drift evaluator (Multi-step Autoregressive Rollout).
Measures error accumulation (drift) against the ground-truth game trajectory across H frames.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class RolloutEvaluator:
    """
    Evaluates autoregressive stability of dynamic models:
        hat_s_{t+1} = f(hat_s_t, a_t)
    and directly compares against ground-truth game trajectories.
    """

    def __init__(
        self,
        device: torch.device,
        max_vx: Optional[float] = None,
        terminal_vy: Optional[float] = None,
        tolerance_px: float = 0.2,
    ):
        """The rollout predicate, with its velocity thresholds optional.

        By default no velocity bound is scored: §10.49/§10.54 record that 72.0 is a speed *class*
        the telemetry never reaches and 64.0 an engine parameter the console leaves (the
        recordings reach 49.0 and 70.0), so neither is a bound a prediction can violate.
        ``velocity_violations`` is then ``None`` rather than a zero that measured nothing.
        The published artifacts scored ``max_vx=72.0`` and ``terminal_vy=64.0``; the pipelines
        that reproduce them pass ``PUBLISHED_MAX_VX`` and ``PUBLISHED_TERMINAL_VY`` by name, and
        10.57 scores the same rollouts against the reachable run class (48.0) as well.
        """
        self.device = device
        self.max_vx = max_vx
        self.terminal_vy = terminal_vy
        self.tolerance_px = tolerance_px

    @property
    def scores_velocity_bounds(self) -> bool:
        """True when at least one velocity threshold was asked for."""
        return self.max_vx is not None or self.terminal_vy is not None

    @torch.no_grad()
    def evaluate_rollout(
        self,
        model: nn.Module,
        model_type: str,
        initial_state: np.ndarray,  # [state_dim]
        action_sequence: np.ndarray,  # [H, action_dim]
        ground_truth_states: np.ndarray,  # [H, state_dim]
    ) -> Dict[str, Any]:
        """
        Executes a complete multi-step autoregressive rollout of length H.

        Returns:
            dict containing:
                predicted_trajectory: [H, state_dim]
                ground_truth_trajectory: [H, state_dim]
                euclidean_drift: [H] (Euclidean distance in pixels at each frame)
                mean_drift: float
                final_drift: float
                kinematic_violations: int
                velocity_violations: int
        """
        model.eval()
        H = len(action_sequence)
        state_dim = len(initial_state)

        pred_traj = np.zeros((H, state_dim), dtype=np.float32)
        curr_state = torch.tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )

        # Buffer for recurrent LSTM if applicable
        if "lstm" in model_type.lower():
            hidden = None

        kin_violations = 0
        vel_violations = 0

        for t in range(H):
            curr_action = torch.tensor(
                action_sequence[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if "lstm" in model_type.lower():
                curr_input_state = curr_state.unsqueeze(1)  # [1, 1, D]
                curr_input_action = curr_action.unsqueeze(1)  # [1, 1, A]
                pred_out, hidden = model(curr_input_state, curr_input_action, hidden)
                next_state_pred = pred_out.squeeze(1)
            else:
                next_state_pred = model(curr_state, curr_action)

            # Check physical violations in predicted step
            hat_vx = next_state_pred[0, 2].item()
            hat_vy = next_state_pred[0, 3].item()
            hat_x = next_state_pred[0, 0].item()
            prev_x = curr_state[0, 0].item()

            # Kinematic violation (> 0.2 pixels deviation from dx = vx/16.0)
            if abs((hat_x - prev_x) - (curr_state[0, 2].item() / 16.0)) > self.tolerance_px:
                kin_violations += 1

            # Velocity threshold violation, scored only against the thresholds asked for:
            # section 4 records 72 and 64 as a speed class and a documented clamp the telemetry
            # leaves, so neither is an absolute bound on real states.
            over_vx = self.max_vx is not None and abs(hat_vx) > self.max_vx
            over_vy = self.terminal_vy is not None and hat_vy > self.terminal_vy
            if over_vx or over_vy:
                vel_violations += 1

            pred_traj[t] = next_state_pred[0].cpu().numpy()
            curr_state = next_state_pred  # Pure open-loop autoregressive feedback

        # Compute Euclidean spatial drift in 2D space (X, Y) in pixels
        gt_xy = ground_truth_states[:, :2]
        pred_xy = pred_traj[:, :2]
        euclidean_drift = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=-1))

        return {
            "predicted_trajectory": pred_traj,
            "ground_truth_trajectory": ground_truth_states,
            "euclidean_drift": euclidean_drift,
            "mean_drift": float(np.mean(euclidean_drift)),
            "final_drift": float(euclidean_drift[-1]),
            "kinematic_violations": kin_violations,
            "velocity_violations": vel_violations if self.scores_velocity_bounds else None,
        }

    @torch.no_grad()
    def evaluate_rollout_multistart(
        self,
        model: nn.Module,
        model_type: str,
        states: np.ndarray,  # [N, state_dim] test states
        actions: np.ndarray,  # [N, action_dim] test actions
        next_states: np.ndarray,  # [N, state_dim] ground-truth next states
        horizon: int = 120,
        num_starts: int = 10,
        stride: int | None = None,
    ) -> Dict[str, Any]:
        """Multi-start open-loop rollout: average drift over spread start indices.

        Single-start rollouts depend heavily on the chosen initial frame. This
        samples ``num_starts`` evenly spaced starts and reports mean ± std of
        mean/final drift plus violation rates, so comparisons carry uncertainty.

        Returns:
            dict with per-start lists (``mean_drifts``, ``final_drifts``,
            ``kinematic_violation_rates``) and aggregates (``mean_drift_mean``,
            ``mean_drift_std``, ``final_drift_mean``, ``final_drift_std``,
            ``kinematic_violation_rate_mean``, ``velocity_violation_rate_mean``,
            ``num_starts``, ``horizon``).
        """
        n = len(actions)
        horizon = min(horizon, n - 1)
        if horizon < 1:
            raise ValueError(f"Need at least 2 transitions for rollout, got {n}")
        num_starts = max(1, min(num_starts, n - horizon))
        if stride is None:
            stride = max(1, (n - horizon) // num_starts)
        starts: List[int] = []
        idx = 0
        while len(starts) < num_starts and idx + horizon <= n:
            starts.append(idx)
            idx += stride
        if len(starts) < num_starts:  # dataset shorter than the even grid
            starts += [n - horizon] * (num_starts - len(starts))

        mean_drifts: List[float] = []
        final_drifts: List[float] = []
        kin_rates: List[float] = []
        vel_rates: List[float] = []
        for s in starts:
            res = self.evaluate_rollout(
                model=model,
                model_type=model_type,
                initial_state=states[s],
                action_sequence=actions[s : s + horizon],
                ground_truth_states=next_states[s : s + horizon],
            )
            mean_drifts.append(float(res["mean_drift"]))
            final_drifts.append(float(res["final_drift"]))
            kin_rates.append(float(res["kinematic_violations"]) / horizon)
            if self.scores_velocity_bounds:
                vel_rates.append(float(res["velocity_violations"]) / horizon)

        return {
            "starts": starts,
            "horizon": horizon,
            "num_starts": len(starts),
            "mean_drifts": mean_drifts,
            "final_drifts": final_drifts,
            "kinematic_violation_rates": kin_rates,
            "velocity_violation_rates": vel_rates if self.scores_velocity_bounds else None,
            "mean_drift_mean": float(np.mean(mean_drifts)),
            "mean_drift_std": float(np.std(mean_drifts, ddof=1)) if len(mean_drifts) > 1 else 0.0,
            "final_drift_mean": float(np.mean(final_drifts)),
            "final_drift_std": float(np.std(final_drifts, ddof=1))
            if len(final_drifts) > 1
            else 0.0,
            "kinematic_violation_rate_mean": float(np.mean(kin_rates)),
            "velocity_violation_rate_mean": (
                float(np.mean(vel_rates)) if self.scores_velocity_bounds else None
            ),
        }

    def benchmark_all_models(
        self,
        models: Dict[str, nn.Module],
        initial_state: np.ndarray,
        action_sequence: np.ndarray,
        ground_truth_states: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        results = {}
        for name, model in models.items():
            model_type = "lstm" if "lstm" in name.lower() else "mlp"
            rollout_res = self.evaluate_rollout(
                model=model,
                model_type=model_type,
                initial_state=initial_state,
                action_sequence=action_sequence,
                ground_truth_states=ground_truth_states,
            )
            results[name] = {
                "mean_drift_pixels": float(rollout_res["mean_drift"]),
                "final_drift_pixels": float(rollout_res["final_drift"]),
                "kin_violations": int(rollout_res["kinematic_violations"]),
                "vel_violations": rollout_res["velocity_violations"],
            }
        return results
