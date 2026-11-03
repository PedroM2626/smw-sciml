"""Tests for aquatic physics parameter identification (Section 10.72).

Guards the empirical findings in results/aquatic_physics_identification_metrics.json:
buoyant gravity reduction, swim stroke impulse, and forward adaptation advantage.
CI-safe (reads the committed metrics artifact; no emulator required).
"""

import json
from pathlib import Path

import pytest

from src.utils.paths import RESULTS_DIR


@pytest.fixture
def aquatic_metrics() -> dict:
    path = Path(RESULTS_DIR) / "aquatic_physics_identification_metrics.json"
    assert path.is_file(), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_aquatic_metrics_has_valid_provenance(aquatic_metrics: dict) -> None:
    assert "_meta" in aquatic_metrics
    meta = aquatic_metrics["_meta"]
    assert meta["command"] == "python -m src.inverse.aquatic_physics_identification"
    assert meta["seed"] == 42


def test_gravity_in_water_is_substantially_reduced_vs_air(aquatic_metrics: dict) -> None:
    g_air = aquatic_metrics["terrestrial_air_regime"]["effective_gravity"]
    g_water = aquatic_metrics["aquatic_fluid_regime"]["effective_gravity"]

    # In air, gravity increment is ~3.0 - 6.0 subpx/f; in water, it is damped by buoyancy to < 1.0 subpx/f
    assert 4.5 < g_air < 7.0, f"Unexpected terrestrial gravity: {g_air}"
    assert 0.1 < g_water < 1.0, f"Unexpected aquatic gravity: {g_water}"
    assert g_water < 0.2 * g_air, "Aquatic gravity must be less than 20% of terrestrial gravity"


def test_aquatic_swim_stroke_impulse_is_distinct_from_jump(aquatic_metrics: dict) -> None:
    j_air = aquatic_metrics["terrestrial_air_regime"]["impulse_strength_subpx"]
    j_water = aquatic_metrics["aquatic_fluid_regime"]["impulse_strength_subpx"]

    # Upward velocities are negative in SMW coordinate conventions
    assert j_air < -70.0, f"Expected strong jump impulse in air: {j_air}"
    assert -35.0 < j_water < -15.0, f"Expected damped swim stroke impulse in water: {j_water}"


def test_kinematic_integration_scale_is_preserved_in_water(aquatic_metrics: dict) -> None:
    scale_water = aquatic_metrics["aquatic_fluid_regime"]["kinematic_scale"]
    res_water = aquatic_metrics["aquatic_fluid_regime"]["max_kinematic_identity_residual"]

    # The 16 subpixels-per-pixel integration rule is a CPU architectural invariant
    assert abs(scale_water - 16.0) < 0.05
    assert res_water < 0.01


def test_aquatic_model_substantially_improves_water_forward_prediction(
    aquatic_metrics: dict,
) -> None:
    transfer = aquatic_metrics["forward_transfer_evaluation"]
    mse_air = transfer["terrestrial_model_on_water"]["y_position_mse_px2"]
    mse_water = transfer["adapted_aquatic_model_on_water"]["y_position_mse_px2"]
    factor = transfer["adaptation_mse_reduction_factor"]

    assert mse_water < mse_air
    assert factor > 5.0, f"Expected at least 5x error reduction, got {factor:.2f}x"
