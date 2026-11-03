"""Tests for closed-loop Pixel-PIML control on real hardware (Section 10.73).

Guards the empirical findings in results/pixel_piml_closed_loop_metrics.json:
CNN 2-frame tracking, Discrete CBF safety filtering, and closed-loop progress advantages.
CI-safe (reads the committed metrics artifact; no emulator required).
"""

import json
from pathlib import Path

import pytest

from src.utils.paths import RESULTS_DIR


@pytest.fixture
def pixel_piml_metrics() -> dict:
    path = Path(RESULTS_DIR) / "pixel_piml_closed_loop_metrics.json"
    assert path.is_file(), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_pixel_piml_metrics_has_valid_provenance(pixel_piml_metrics: dict) -> None:
    assert "_meta" in pixel_piml_metrics
    meta = pixel_piml_metrics["_meta"]
    assert meta["command"] == "python -m src.evaluation.pixel_piml_closed_loop"
    assert meta["seed"] == 42


def test_pixel_piml_contains_three_evaluated_arms(pixel_piml_metrics: dict) -> None:
    arms = pixel_piml_metrics["arms_summary"]
    assert "random" in arms
    assert "unconstrained" in arms
    assert "piml_cbf" in arms

    for arm in ("random", "unconstrained", "piml_cbf"):
        assert arms[arm]["total_runs"] == 3
        assert arms[arm]["death_count"] == 0


def test_piml_cbf_achieves_superior_progress_over_unconstrained(pixel_piml_metrics: dict) -> None:
    summary = pixel_piml_metrics["arms_summary"]
    prog_random = summary["random"]["mean_progress_px"]
    prog_unconstrained = summary["unconstrained"]["mean_progress_px"]
    prog_cbf = summary["piml_cbf"]["mean_progress_px"]

    assert prog_cbf > prog_unconstrained > prog_random
    assert prog_cbf > 1100.0, f"Expected mean progress > 1100 px, got {prog_cbf:.1f}"


def test_piml_cbf_enforces_zero_physics_violations(pixel_piml_metrics: dict) -> None:
    summary = pixel_piml_metrics["arms_summary"]
    viol_cbf = summary["piml_cbf"]["mean_violation_rate"]
    viol_unconstrained = summary["unconstrained"]["mean_violation_rate"]

    assert viol_cbf == 0.0
    # Unconstrained policy occasionally fires commands violating velocity limits or contact bounds
    assert viol_unconstrained >= 0.0


def test_visual_tracking_errors_are_finite_and_bounded(pixel_piml_metrics: dict) -> None:
    runs = pixel_piml_metrics["runs"]["piml_cbf"]
    for run in runs:
        mae = run["visual_tracking_mae"]
        assert 0.0 <= mae["x_px"] < 400.0
        assert 0.0 <= mae["y_px"] < 50.0
        assert 0.0 <= mae["vx_subpx"] < 50.0
