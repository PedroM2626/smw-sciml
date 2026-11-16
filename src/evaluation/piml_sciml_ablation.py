"""
piml_sciml_ablation.py
What does the physics-informed part of PIML-PPO contribute? An ablation (README 10.75).

10.71 and 10.74 report closed-loop clears by `PIML-PPO` and credit them to physics-informed
reinforcement learning, but every such run trains with a demonstration anchor - a cross-entropy
term on the reference trajectory in every PPO update and a final polish until the network agrees
with every anchor action - and no arm trained without the physics-informed terms exists. This
study trains four arms on the same stages, seeds and budget:

* `full`                  - as shipped: CBF logit filter, HJB critic term, action-physics term,
                            Hard PINN head, self-imitation, polish.
* `no_sciml`              - the same with the four physics-informed terms switched off
                            (`cbf_beta = 0`, `lambda_hjb = lambda_phys_act = lambda_pinn = 0`).
* `no_imitation`          - SciML on, but no self-imitation term and no polish: what PPO with
                            frontier resets and the physics-informed terms does by itself.
* `no_imitation_no_sciml` - neither.

Two things stay in every arm because they are the demonstration entering by another door: the
frontier resets (rollouts start from savestates along the reference trajectory) and the value
anchor (the critic is also fitted to the reference trajectory's returns). The study says what that
leaves it able to conclude. Each run is evaluated in closed loop from the stage's first frame twice:
right after the PPO epochs and again after the polish.

Needs the console (core + ROM) and the stages' savestates. Run:  python -m src.evaluation.piml_sciml_ablation
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import Any, Dict, List, Sequence

from src.utils.logging import get_logger
from src.utils.paths import RESULTS_DIR
from src.utils.provenance import write_metrics

logger = get_logger(__name__)

STAGES = (
    "smw_map_node_68_135_t29.state",
    "smw_map_round_1_152_104_t27.state",
    "smw_map_round_2_184_72_t26.state",
)
STAGE_LABEL = {"t29": "`0x29`", "t27": "`0x27`", "t26": "`0x26`"}
SEEDS = (42, 43, 44)

NO_SCIML: Dict[str, Any] = {
    "cbf_beta": 0.0,
    "lambda_hjb": 0.0,
    "lambda_phys_act": 0.0,
    "lambda_pinn": 0.0,
}
NO_IMITATION: Dict[str, Any] = {"lambda_sil": 0.0, "polish": False}

ARMS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_sciml": dict(NO_SCIML),
    "no_imitation": dict(NO_IMITATION),
    "no_imitation_no_sciml": {**NO_SCIML, **NO_IMITATION},
}


def stage_tag(state_name: str) -> str:
    return os.path.splitext(os.path.basename(state_name))[0][-3:]


def _one_run(job: Dict[str, Any]) -> Dict[str, Any]:
    """Train one (stage, arm, seed) cell in its own process: the core holds process-global state."""
    from src.training.piml_stage_ppo import train_piml_stage_ppo_single

    result = train_piml_stage_ppo_single(
        job["state"],
        epochs=job["epochs"],
        seed=job["seed"],
        device_str=job["device"],
        save_checkpoint=False,
        require_clear=False,
        pre_polish_eval=True,
        **ARMS[job["arm"]],
    )
    final = result["closed_loop_evaluation"]
    diag = result["training_diagnostics"]
    pre = diag["pre_polish_evaluation"] or final
    return {
        "stage": job["state"],
        "arm": job["arm"],
        "seed": job["seed"],
        "after_ppo": {k: pre[k] for k in ("cleared", "died", "progress_px", "frames")},
        "final": {k: final[k] for k in ("cleared", "died", "progress_px", "frames")},
        "polish_steps": diag["polish_steps"],
        "anchor_accuracy_after_ppo": diag["anchor_accuracy_after_ppo"],
        "policy_action_agreement": result["sciml_diagnostics"]["policy_action_agreement"],
    }


def summarise(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per stage and arm: how many seeds cleared after the PPO epochs and after the polish."""
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stage in sorted({r["stage"] for r in runs}):
        table[stage] = {}
        for arm in ARMS:
            cell = [r for r in runs if r["stage"] == stage and r["arm"] == arm]
            if not cell:
                continue
            n = len(cell)
            table[stage][arm] = {
                "seeds": n,
                "cleared_after_ppo": sum(bool(r["after_ppo"]["cleared"]) for r in cell),
                "cleared_final": sum(bool(r["final"]["cleared"]) for r in cell),
                "mean_progress_after_ppo": sum(r["after_ppo"]["progress_px"] for r in cell) / n,
                "mean_progress_final": sum(r["final"]["progress_px"] for r in cell) / n,
                "mean_polish_steps": sum(r["polish_steps"] for r in cell) / n,
                "mean_anchor_accuracy_after_ppo": sum(r["anchor_accuracy_after_ppo"] for r in cell)
                / n,
            }
    return table


def render_table(summary: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    """The README 10.75 table rows, generated from the artifact so the prose cannot drift from it."""
    rows: List[str] = []
    for stage, arms in summary.items():
        for arm, cell in arms.items():
            rows.append(
                f"| {STAGE_LABEL.get(stage_tag(stage), stage)} | `{arm}` | "
                f"{cell['cleared_after_ppo']:.0f} / {cell['seeds']:.0f} | "
                f"{cell['mean_progress_after_ppo']:.1f} px | "
                f"{cell['cleared_final']:.0f} / {cell['seeds']:.0f} | "
                f"{cell['mean_progress_final']:.1f} px | "
                f"{cell['mean_polish_steps']:.0f} |"
            )
    return rows


def run_ablation(
    stages: Sequence[str] = STAGES,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[str] = tuple(ARMS),
    epochs: int = 25,
    device: str = "cuda",
    output_dir: str = RESULTS_DIR,
) -> Dict[str, Any]:
    jobs = [
        {"state": stage, "arm": arm, "seed": seed, "epochs": epochs, "device": device}
        for stage in stages
        for arm in arms
        for seed in seeds
    ]
    runs: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    for job in jobs:
        with ctx.Pool(1, maxtasksperchild=1) as pool:
            run = pool.apply(_one_run, (job,))
        runs.append(run)
        logger.info(
            "%s %-22s seed %d | after PPO: cleared=%s %.0f px | final: cleared=%s %.0f px",
            stage_tag(run["stage"]),
            run["arm"],
            run["seed"],
            run["after_ppo"]["cleared"],
            run["after_ppo"]["progress_px"],
            run["final"]["cleared"],
            run["final"]["progress_px"],
        )
    payload = {
        "study": "What the physics-informed terms of PIML-PPO contribute: a four-arm ablation",
        "protocol": {
            "stages": list(stages),
            "arms": {name: overrides for name, overrides in ARMS.items() if name in arms},
            "seeds": list(seeds),
            "epochs": epochs,
            "kept_in_every_arm": [
                "frontier resets along the reference trajectory",
                "value anchor on the reference trajectory's returns",
            ],
            "evaluation": "deterministic closed loop from the stage's first frame, after the PPO "
            "epochs and again after the polish, under the stage's message-box pulse",
        },
        "runs": runs,
        "summary": summarise(runs),
    }
    write_metrics(
        os.path.join(output_dir, "piml_sciml_ablation_metrics.json"),
        payload,
        seed=int(seeds[0]),
        command="python -m src.evaluation.piml_sciml_ablation",
    )
    return payload


def main() -> None:
    from src.utils.config import parse_args_with_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default=None)
    parser.add_argument("--stages", nargs="*", default=list(STAGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parse_args_with_config(parser)
    run_ablation(args.stages, args.seeds, args.arms, args.epochs, args.device, args.output_dir)


if __name__ == "__main__":
    main()
