r"""
conformal_uncertainty.py
Distribution-free prediction intervals for the identified laws (README 10.56.1).

Section 10.43 measures the seed-to-seed spread of a discovered law because a tree is not a
differentiable parameterisation: "how uncertain is this discovered law?" cannot be answered
with the Laplace machinery of 10.40 - no Hessian, no Fisher spectrum - and the honest
substitute published there is a spread over three search seeds. Split conformal prediction
is the missing instrument: it asks nothing of the law's functional form, needs only a
calibration split the fit never saw, and returns intervals with a finite-sample marginal
guarantee - coverage of at least 1 - alpha on any exchangeable draw, whatever the residual
distribution is. Section 10.58 already measured why the parametric alternative fails on
this telemetry (the residue has memory and a tail an iid Gaussian under-predicts by up to
5.95x at 60 frames), so the Gaussian arm is carried here as the comparator that measurement
says will under-cover.

The estimator (Vovk, Gammerman and Shafer 2005; Papadopoulos et al. 2002 for the split
variant) is three lines of arithmetic, which is the point:

1. score every calibration row by its absolute residual under the fitted law;
2. take the ceil((n+1)(1-alpha))-th smallest score as the half-width q;
3. issue [prediction - q, prediction + q] on every new row.

No smoothness, no symmetry and no boundedness is assumed of the residuals, and the
order statistic - not the empirical quantile - is what makes the guarantee finite-sample
rather than asymptotic: with n calibration scores, the (n+1)-st draw is ranked among them
uniformly at random under exchangeability, so P(new score <= q) >= 1 - alpha exactly.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import norm


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The split-conformal half-width: the k-th smallest calibration score.

    k = ceil((n + 1) * (1 - alpha)) is the smallest order statistic that makes the
    marginal coverage guarantee hold exactly rather than asymptotically. When k
    exceeds n the calibration split is too small to exclude anything at this alpha,
    and the honest half-width is infinite - returned as math.inf so the artifact
    writer (which refuses non-finite JSON) can translate it into an explicit null.
    """
    ordered = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    n = ordered.size
    if n == 0:
        return math.inf
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return math.inf
    return float(ordered[k - 1])


def gaussian_half_width(residuals: np.ndarray, alpha: float) -> float:
    """The iid-Gaussian half-width z_(1-alpha/2) * sigma of the calibration residuals.

    The comparator arm. A Gaussian interval is the uncertainty statement a
    least-squares fit implies, and it is wrong on this data for the reason 10.58
    measured: the residue is a process with memory and heavy tails, not iid noise.
    Carrying it is what makes a conformal number readable instead of
    self-congratulatory - the interesting row is the one where the two disagree.
    """
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if residuals.size < 2:
        return math.inf
    sigma = float(residuals.std(ddof=1))
    return float(norm.ppf(1.0 - alpha / 2.0) * sigma)


def mean_null_quantile(truth: np.ndarray, alpha: float) -> float:
    """The fit-mean null: the conformal half-width around the calibration mean.

    Every per-law table in this repository is reported against the predictor that
    ignores the law, and an uncertainty claim needs the same control: a law whose
    interval is no tighter than the mean's has earned nothing, and only the null
    makes that visible.
    """
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if truth.size == 0:
        return math.inf
    centre = float(truth.mean())
    return conformal_quantile(np.abs(truth - centre), alpha)


def coverage_rate(prediction: np.ndarray, truth: np.ndarray, half_width: float) -> float:
    """The fraction of evaluation rows inside [prediction - q, prediction + q].

    An unbounded interval (q infinite, the small-calibration case) contains every row
    by construction and is reported as coverage 1.0 with the unbounded flag set
    beside it, so a reader cannot mistake a too-small calibration split for a
    good law.
    """
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if truth.size == 0:
        return float("nan")
    if not math.isfinite(half_width):
        return 1.0
    return float(np.mean(np.abs(truth - prediction) <= half_width + 1e-12))
