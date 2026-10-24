r"""
conformal_uncertainty_benchmark.py
Distribution-free intervals on the laws 10.56 identifies (README 10.56.1).

Section 10.43 publishes the seed-to-seed spread of a discovered law as the honest
substitute for the Bayesian uncertainty it cannot compute, because a regression tree
has no Hessian to invert. Split conformal prediction needs no Hessian either: it
scores the fitted law on a calibration split the fit never saw, reads the half-width
off an order statistic, and guarantees marginal coverage of at least 1 - alpha on any
exchangeable draw - a finite-sample statement, with no assumption on the residual
distribution at all.

This study applies it to the SINDy laws of 10.56, exactly as that section fits them
(Huber loss, alpha = 0.05, degree-2 dictionary, duplicate columns pruned), on every
recording this repository has:

* the **law arm**: intervals around the identified law's own prediction;
* the **Gaussian arm**: z * sigma of the same calibration residuals, the parametric
  statement 10.58 already measured under-predicting this telemetry's tails;
* the **fit-mean null**: the same conformal machinery centred on the calibration
  mean, the control every per-law table in this repository carries.

Calibration is the canonical validation split and evaluation the canonical test
split of load_and_preprocess_data, so the rows being scored are exactly the ones the
10.47/10.53/10.55 grids trained and scored on - no new protocol to defend.

Writes results/conformal_uncertainty_metrics.json. Emulator-free, about a minute.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.environment.dataset_loader import load_and_preprocess_data
from src.inverse.conformal_uncertainty import (
    conformal_quantile,
    coverage_rate,
    gaussian_half_width,
    mean_null_quantile,
)
from src.inverse.sindy_identification import (
    DYNAMIC_TARGETS,
    evaluate_library,
    increments,
    library_inputs_of,
    library_terms,
    prune_columns,
    sindy,
    term_names,
)
from src.utils.config import parse_args_with_config
from src.utils.logging import get_logger
from src.utils.paths import (
    DATASET_GAMEPLAY,
    DATASET_JUMP,
    DATASET_MULTI_ENTITY,
    DATASET_SPRINT,
    DATASET_TILEMAP,
    RESULTS_DIR,
)
from src.utils.provenance import write_metrics
from src.utils.typography import demath_typographic

logger = get_logger(__name__)

ARTIFACT_NAME = "conformal_uncertainty_metrics.json"

RECORDINGS: Tuple[Tuple[str, str], ...] = (
    ("published_gameplay", DATASET_GAMEPLAY),
    ("tilemap", DATASET_TILEMAP),
    ("multi_entity", DATASET_MULTI_ENTITY),
    ("sprint_targeted", DATASET_SPRINT),
    ("jump_targeted", DATASET_JUMP),
)

DISPLAY = {
    "published_gameplay": "published gameplay",
    "tilemap": "tilemap (10.41)",
    "multi_entity": "multi-entity (10.31)",
    "sprint_targeted": "sprint-targeted (10.45)",
    "jump_targeted": "jump-targeted (10.52)",
}

#: miscoverage levels; the README table quotes the 0.10 row and the artifact carries all three
ALPHAS: Tuple[float, ...] = (0.05, 0.10, 0.20)
SELECTED_ALPHA = 0.10
#: the SINDy recipe of 10.56, unchanged: robust loss, this threshold, degree-2 dictionary
SINDY_ALPHA = 0.05
MAX_DEGREE = 2


def _fit_and_calibrate(
    label: str,
    path: str,
    seed: int,
    alphas: Sequence[float],
    sindy_alpha: float,
    max_degree: int,
) -> Optional[Dict[str, Any]]:
    """Fit the 10.56 law on train, calibrate on val, score every arm on test."""
    if not os.path.isfile(path):
        logger.warning("%s missing; the %s row is omitted.", path, label)
        return None
    data = load_and_preprocess_data(dataset_path=path, seed=seed)
    terms = library_terms(max_degree)
    names = term_names(terms)

    def pack(split: str) -> Tuple[np.ndarray, np.ndarray]:
        inputs = library_inputs_of(data[f"{split}_states"], data[f"{split}_actions"])
        theta = evaluate_library(inputs, terms)
        xdot = increments(data[f"{split}_states"], data[f"{split}_next_states"])[
            :, : len(DYNAMIC_TARGETS)
        ]
        return theta, xdot

    theta_train, xdot_train, _ = pack("train")
    theta_train, kept_names, dropped, keep = prune_columns(theta_train, names)
    fit = sindy(
        xdot_train,
        theta_train,
        kept_names,
        alpha=sindy_alpha,
        targets=list(DYNAMIC_TARGETS),
        robust=True,
    )

    predictions: Dict[str, np.ndarray] = {}
    truths: Dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        theta_full, xdot, _ = pack(split)
        predictions[split] = theta_full[:, keep] @ fit.coefficients
        truths[split] = xdot

    out: Dict[str, Any] = {
        "dataset": os.path.basename(path),
        "n_train_transitions": int(theta_train.shape[0]),
        "n_calibration_transitions": int(truths["val"].shape[0]),
        "n_test_transitions": int(truths["test"].shape[0]),
        "kept_terms": len(kept_names),
        "dropped_duplicates": dropped,
        "targets": {},
    }
    for j, target in enumerate(DYNAMIC_TARGETS):
        calib_pred = predictions["val"][:, j]
        calib_truth = truths["val"][:, j]
        eval_pred = predictions["test"][:, j]
        eval_truth = truths["test"][:, j]
        scores = np.abs(calib_truth - calib_pred)
        residuals = calib_truth - calib_pred
        centre = float(calib_truth.mean()) if calib_truth.size else 0.0
        per_alpha: Dict[str, Dict[str, Any]] = {}
        for alpha in alphas:
            q = conformal_quantile(scores, alpha)
            z_sigma = gaussian_half_width(residuals, alpha)
            q_null = mean_null_quantile(calib_truth, alpha)
            per_alpha[f"{alpha:g}"] = {
                "half_width": None if math.isinf(q) else round(q, 6),
                "unbounded": bool(math.isinf(q)),
                "coverage": round(coverage_rate(eval_pred, eval_truth, q), 6),
                "gaussian_half_width": round(z_sigma, 6),
                "gaussian_coverage": round(coverage_rate(eval_pred, eval_truth, z_sigma), 6),
                "null_centre": round(centre, 6),
                "null_half_width": None if math.isinf(q_null) else round(q_null, 6),
                "null_coverage": round(
                    coverage_rate(np.full_like(eval_truth, centre), eval_truth, q_null), 6
                ),
            }
        out["targets"][target] = {"alphas": per_alpha}
    return out


def _verdict(results: Dict[str, Any], alphas: Sequence[float]) -> Dict[str, Any]:
    """Coverage per miscoverage level: does each arm reach the target it claims?"""
    cells: List[Tuple[str, str, float, Dict[str, Any]]] = []
    for label, block in results.items():
        for target, tblock in block["targets"].items():
            for key, ablock in tblock["alphas"].items():
                cells.append((label, target, float(key), ablock))

    verdict: Dict[str, Any] = {}
    for alpha in alphas:
        key = f"{alpha:g}"
        selected = [cell for cell in cells if cell[2] == alpha]
        target_cov = round(1.0 - alpha, 10)
        if not selected:
            verdict[key] = {"target_coverage": target_cov, "cells": 0}
            continue
        bounded = [cell for cell in selected if not cell[3]["unbounded"]]
        conformal_covs = [cell[3]["coverage"] for cell in bounded]
        width_ratios = [
            cell[3]["gaussian_half_width"] / cell[3]["half_width"]
            for cell in bounded
            if cell[3]["half_width"]
        ]
        verdict[key] = {
            "target_coverage": target_cov,
            "cells": len(selected),
            "unbounded_cells": len(selected) - len(bounded),
            "conformal_cells_at_or_above_target": sum(
                cov >= target_cov - 1e-9 for cov in conformal_covs
            ),
            "conformal_min_coverage": min(conformal_covs) if conformal_covs else None,
            "gaussian_cells_below_target": sum(
                cell[3]["gaussian_coverage"] < target_cov for cell in selected
            ),
            "gaussian_min_coverage": min(cell[3]["gaussian_coverage"] for cell in selected),
            "null_min_coverage": min(cell[3]["null_coverage"] for cell in selected),
            "median_gaussian_over_conformal_width": (
                float(np.median(width_ratios)) if width_ratios else None
            ),
        }

    if not cells:
        verdict["reading"] = "no recordings were available; nothing was calibrated"
        return verdict
    headline = verdict[f"{SELECTED_ALPHA:g}"]
    verdict["reading"] = (
        f"Split conformal intervals on the 10.56 laws, calibrated on the canonical "
        f"validation split and scored on the test split. At the "
        f"{1 - SELECTED_ALPHA:.0%} level the law arm covers at or above target on "
        f"{headline['conformal_cells_at_or_above_target']} of "
        f"{headline['cells'] - headline['unbounded_cells']} bounded cells "
        f"(minimum {headline['conformal_min_coverage']:.3f}), while the iid-Gaussian "
        f"arm is below its own {1 - SELECTED_ALPHA:.0%} claim on "
        f"{headline['gaussian_cells_below_target']} of {headline['cells']} cells "
        f"(minimum {headline['gaussian_min_coverage']:.3f}) - the same tail "
        f"under-prediction 10.58 measured for the residue, now on the identified "
        f"laws' own predictions."
    )
    return verdict


def render_coverage_table(results: Dict[str, Any], alpha: float = SELECTED_ALPHA) -> List[str]:
    """One row per recording and target at the selected level, all three arms."""
    key = f"{alpha:g}"
    rows: List[str] = []
    for label, block in results.items():
        for target, tblock in block["targets"].items():
            arm = tblock["alphas"][key]
            width = "unbounded" if arm["unbounded"] else f"{arm['half_width']:.4f}"
            rows.append(
                f"| {DISPLAY.get(label, label)} | {target} | {arm['coverage']:.3f}"
                f" | {width} | {arm['gaussian_coverage']:.3f}"
                f" | {arm['null_coverage']:.3f} |"
            )
    return [demath_typographic(row) for row in rows]


def run_conformal_uncertainty_benchmark(
    output_dir: Optional[str] = None,
    seed: int = 42,
    alphas: Sequence[float] = ALPHAS,
    sindy_alpha: float = SINDY_ALPHA,
    max_degree: int = MAX_DEGREE,
    recordings: Sequence[Tuple[str, str]] = RECORDINGS,
) -> Dict[str, Any]:
    """Conformal intervals on every recording, against the Gaussian and mean-null arms."""
    out_dir = output_dir or RESULTS_DIR
    logger.info("=== Conformal uncertainty on %d recordings ===", len(recordings))
    results: Dict[str, Any] = {}
    for label, path in recordings:
        block = _fit_and_calibrate(
            label,
            path,
            seed=seed,
            alphas=alphas,
            sindy_alpha=sindy_alpha,
            max_degree=max_degree,
        )
        if block is None:
            continue
        results[label] = block
        row = block["targets"]["dx"]["alphas"][f"{SELECTED_ALPHA:g}"]
        logger.info(
            "  %-18s | n_calib %5d | dx q %.4f | coverage %.3f | gaussian %.3f",
            label,
            block["n_calibration_transitions"],
            row["half_width"] if row["half_width"] is not None else float("inf"),
            row["coverage"],
            row["gaussian_coverage"],
        )

    alpha_list = list(alphas)
    payload: Dict[str, Any] = {
        "study": (
            "Split conformal prediction on the SINDy laws of 10.56: finite-sample "
            "marginal coverage on the canonical validation/test splits, against the "
            "iid-Gaussian arm and the fit-mean null."
        ),
        "protocol": {
            "seed": seed,
            "alphas": alpha_list,
            "selected_alpha": SELECTED_ALPHA,
            "law": "SINDy (10.56): Huber loss, degree-2 dictionary, duplicate columns pruned",
            "sindy_alpha": sindy_alpha,
            "max_degree": max_degree,
            "calibration_split": "canonical validation episodes",
            "evaluation_split": "canonical test episodes",
            "targets": list(DYNAMIC_TARGETS),
            "emulator_required": False,
        },
        "results": results,
        "verdict": _verdict(results, alpha_list),
    }

    artifact = os.path.join(out_dir, ARTIFACT_NAME)
    write_metrics(
        artifact,
        payload,
        seed=seed,
        command=(
            "python -m src.evaluation.conformal_uncertainty_benchmark"
            f" --alphas {','.join(f'{a:g}' for a in alpha_list)}"
        ),
    )
    logger.info("Metrics written to %s", artifact)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Split conformal intervals on the 10.56 identified laws (10.56.1): "
            "finite-sample coverage against the Gaussian and mean-null arms."
        )
    )
    parser.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        default=",".join(f"{a:g}" for a in ALPHAS),
        help="comma-separated miscoverage levels (default 0.05,0.1,0.2)",
    )
    parser.add_argument("--sindy-alpha", type=float, default=SINDY_ALPHA)
    parser.add_argument("--max-degree", type=int, default=MAX_DEGREE)
    args = parse_args_with_config(parser)

    run_conformal_uncertainty_benchmark(
        output_dir=args.output_dir,
        seed=args.seed,
        alphas=[float(a) for a in str(args.alphas).split(",")],
        sindy_alpha=args.sindy_alpha,
        max_degree=args.max_degree,
    )
