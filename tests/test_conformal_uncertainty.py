r"""Tests for split conformal prediction on the identified laws (README 10.56.1).

Unit level: the finite-sample order statistic, the marginal guarantee on
exchangeable data, the Gaussian arm's predicted failure on heavy tails, and the
fit-mean control. Artifact level: the README 10.56.1 table rows and headline
numbers are generated from the committed artifact, so the prose cannot drift
from the measurement the way a pasted table can.

Run:  pytest tests/test_conformal_uncertainty.py -q
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.conformal_uncertainty_benchmark import (
    ARTIFACT_NAME,
    render_coverage_table,
)
from src.inverse.conformal_uncertainty import (
    conformal_quantile,
    coverage_rate,
    gaussian_half_width,
    mean_null_quantile,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / ARTIFACT_NAME
README = REPO / "README.md"


# ---------- the order statistic ----------


def test_quantile_is_the_finite_sample_order_statistic():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    # n=5, alpha=0.2: k = ceil(6 * 0.8) = 5 -> the largest calibration score.
    assert conformal_quantile(scores, 0.2) == 5.0
    # n=5, alpha=0.5: k = ceil(6 * 0.5) = 3.
    assert conformal_quantile(scores, 0.5) == 3.0


def test_quantile_is_infinite_when_calibration_cannot_rule_anything_out():
    # n=9, alpha=0.05: k = ceil(10 * 0.95) = 10 > 9 - the honest interval is everything.
    assert math.isinf(conformal_quantile(list(range(9)), 0.05))
    assert math.isinf(conformal_quantile([], 0.1))


def test_unbounded_interval_covers_everything():
    truth = np.array([100.0, -100.0, 5.0, 0.0])
    assert coverage_rate(np.zeros(4), truth, math.inf) == 1.0


def test_coverage_counts_rows_inside_the_band():
    pred = np.zeros(5)
    truth = np.array([0.0, 0.9, 1.0, 1.1, 5.0])
    assert coverage_rate(pred, truth, 1.0) == pytest.approx(3.0 / 5.0)


# ---------- the guarantee, and the arms that fail it ----------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_coverage_guarantee_holds_on_exchangeable_data(alpha):
    rng = np.random.default_rng(42)
    calibration = np.abs(rng.normal(size=4000))
    evaluation = np.abs(rng.normal(size=8000))
    q = conformal_quantile(calibration, alpha)
    covered = float(np.mean(evaluation <= q))
    assert covered >= 1.0 - alpha - 0.01


def test_gaussian_width_pays_for_heavy_tails_in_width_conformal_does_not():
    """On a heavy-tailed residual the Gaussian interval prepays in width, not in guarantee.

    A lognormal residual inflates the sample sigma the Gaussian arm reads its width
    from, so the parametric interval covers by being more than one and a half times
    as wide as the conformal one (seed 7 measures z = 10.78 against q = 6.47) - and
    it still carries no finite-sample guarantee. The conformal
    half-width is the order statistic: it hits the target coverage with the width
    the residual distribution actually needs. Which arm is *wider* on the real
    telemetry is a measurement, reported by the artifact; that it needs no shape
    assumption is the property.
    """
    rng = np.random.default_rng(7)
    calibration = rng.lognormal(0.0, 1.5, size=3000)
    q = conformal_quantile(calibration, 0.1)
    z = gaussian_half_width(calibration, 0.1)
    evaluation = rng.lognormal(0.0, 1.5, size=6000)
    conformal_covered = float(np.mean(evaluation <= q))
    gaussian_covered = float(np.mean(evaluation <= z))
    assert conformal_covered >= 0.9 - 0.01
    assert gaussian_covered >= 0.9  # it covers here - by being far too wide
    assert z > 1.5 * q


def test_mean_null_is_wider_than_a_good_law():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2000, 3))
    truth = x @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.normal(size=2000)
    prediction = x @ np.array([1.0, -2.0, 0.5])
    law_q = conformal_quantile(np.abs(truth - prediction)[:1000], 0.1)
    null_q = mean_null_quantile(truth[:1000], 0.1)
    assert law_q < null_q


# ---------- the artifact gates ----------


def _payload() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip(f"{ARTIFACT_NAME} not generated yet")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_carries_provenance():
    payload = _payload()
    meta = payload.get("_meta")
    assert isinstance(meta, dict) and "command" in meta and "seed" in meta


def test_readme_coverage_table_rows_are_generated_from_the_artifact():
    payload = _payload()
    rows = render_coverage_table(payload["results"])
    assert rows, "the coverage table rendered no rows"
    text = README.read_text(encoding="utf-8")
    missing = [row for row in rows if row not in text]
    assert not missing, (
        "README 10.56.1 no longer quotes the artifact rows it claims to be "
        f"generated from: {missing}"
    )


def test_readme_quotes_the_verdict_of_the_selected_level():
    payload = _payload()
    verdict = payload["verdict"]["0.1"]
    text = README.read_text(encoding="utf-8")
    bounded = verdict["cells"] - verdict["unbounded_cells"]
    quoted = [
        f"{verdict['conformal_cells_at_or_above_target']} of {bounded}",
        f"minimum coverage {verdict['conformal_min_coverage']:.3f}",
        f"{verdict['gaussian_cells_below_target']} of {verdict['cells']}",
        f"minimum {verdict['gaussian_min_coverage']:.3f}",
    ]
    for fragment in quoted:
        assert fragment in text, (
            f"README 10.56.1 no longer quotes {fragment!r} from the verdict block of "
            f"{ARTIFACT_NAME}"
        )


def test_every_bounded_cell_meets_the_guarantee_within_sampling_noise():
    """The marginal guarantee, checked per cell: 2 points of slack for a finite test split.

    The guarantee is marginal over exchangeable draws, so a single cell may dip below
    target by sampling noise - but a cell far below it would mean the calibration and
    evaluation splits are not exchangeable, which is exactly what this gate exists
    to notice.
    """
    payload = _payload()
    worst = 1.0
    for block in payload["results"].values():
        for target_block in block["targets"].values():
            arm = target_block["alphas"]["0.1"]
            if arm["unbounded"]:
                continue
            worst = min(worst, arm["coverage"])
    assert worst >= 0.9 - 0.02, (
        f"a bounded 90% cell covers only {worst:.3f}: the calibration and evaluation "
        "splits disagree more than sampling noise explains"
    )
