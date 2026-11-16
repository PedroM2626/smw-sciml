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


# --- the identification itself, on frames generated from laws we chose ----------------------------


def _frames(gravity: float, impulse: float, water: bool, n: int = 600, seed: int = 0) -> dict:
    """Synthetic transitions: kinematic identity dx = vx / 16, constant gravity, a swim/jump impulse."""
    import numpy as np

    rng = np.random.default_rng(seed)
    vx = rng.uniform(-30.0, 30.0, n)
    vy = rng.uniform(0.0, 20.0, n)
    button_b = (rng.random(n) < 0.25).astype(float)
    next_vy = np.where(button_b > 0.5, impulse, vy + gravity)
    y = rng.uniform(100.0, 300.0, n)
    return {
        "x": rng.uniform(0.0, 1000.0, n),
        "vx": vx,
        "next_vx": 0.98 * vx,
        "vy": vy,
        "next_vy": next_vy,
        "y": y,
        "next_y": y + next_vy / 16.0,
        "c_ground": np.zeros(n),
        "button_b": button_b,
        "in_water": np.full(n, 1.0 if water else 0.0),
    } | {"next_x": None}


def _with_next_x(data: dict) -> dict:
    data["next_x"] = data["x"] + data["vx"] / 16.0
    return data


def test_the_identification_recovers_the_laws_the_frames_were_generated_from() -> None:
    import numpy as np

    from src.inverse.aquatic_physics_identification import identify_regime_parameters

    air = _with_next_x(_frames(6.0, -32.0, water=False))
    fit = identify_regime_parameters(air, np.ones(600, dtype=bool))
    assert fit["effective_gravity"] == pytest.approx(6.0, abs=1e-6)
    assert fit["impulse_strength_subpx"] == pytest.approx(-32.0, abs=1e-6)
    assert fit["kinematic_scale"] == pytest.approx(16.0, rel=1e-6)
    assert fit["max_kinematic_identity_residual"] == pytest.approx(0.0, abs=1e-6)
    assert fit["velocity_retention_ratio"] == pytest.approx(0.98, abs=1e-6)
    assert fit["n_samples"] == 600


def test_the_identification_degrades_to_neutral_values_on_too_little_data() -> None:
    import numpy as np

    from src.inverse.aquatic_physics_identification import identify_regime_parameters

    data = _with_next_x(_frames(6.0, -32.0, water=False, n=600))
    fit = identify_regime_parameters(data, np.arange(600) < 3)
    assert fit["effective_gravity"] == 0.0 and fit["impulse_strength_subpx"] == 0.0
    assert fit["kinematic_scale"] == 16.0 and fit["velocity_retention_ratio"] == 1.0


def test_an_adapted_aquatic_model_beats_the_terrestrial_one_on_water_frames() -> None:
    import numpy as np

    from src.inverse.aquatic_physics_identification import (
        evaluate_forward_regime_transfer,
        identify_regime_parameters,
    )

    air = _with_next_x(_frames(6.0, -32.0, water=False, seed=1))
    water = _with_next_x(_frames(0.5, -24.0, water=True, seed=2))
    terrestrial = identify_regime_parameters(air, np.ones(600, dtype=bool))
    aquatic = identify_regime_parameters(water, np.ones(600, dtype=bool))
    report = evaluate_forward_regime_transfer(water, terrestrial, aquatic)
    assert report["n_evaluated_water_transitions"] == 600
    assert report["adaptation_mae_reduction_factor"] > 1.5
    assert (
        report["adapted_aquatic_model_on_water"]["y_position_mae_px"]
        < report["terrestrial_model_on_water"]["y_position_mae_px"]
    )
    with pytest.raises(AssertionError, match="Insufficient water transitions"):
        evaluate_forward_regime_transfer(air, terrestrial, aquatic)
