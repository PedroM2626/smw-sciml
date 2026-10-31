[← 10.37](10-37-analytical-and-oracle-model-baselines.md) · [Contents](../../README.md#contents) · [10.39 →](10-39-physics-informed-model-free-rl-piml-mfrl.md)

### 10.38 Closed-Loop Reproduction Audit and Preamble Probe

Refreshing §10.6 exposed a methodological hole that no unit test could see: two
harnesses in this repository reported 164.8 px and 573.9 px for *the same
checkpoint, planner and objective*. `src/evaluation/mbrl_mpc_benchmark.py` now ships
two diagnostics that turn that into measurable statements.

#### 10.38.1 Preamble Probe (`--preamble-probe`)

Only the episode start varies; model, objective, CEM budget and seed are fixed
(`results/mpc_preamble_probe_metrics.json`):

| Episode preamble | Mode at first planned frame | Progress | Survived | Alignment error |
| :--- | :---: | :---: | :---: | :---: |
| restore only *(the original §10.6 harness)* | `0x08` | 168.81 px | 300 | 3.70 px |
| restore + warm-up frames | `0x08` | 165.31 px | 300 | 3.94 px |
| restore + force gameplay mode | `0x14` | 117.19 px | 174 | 1.54 px |
| **restore + mode + warm-up (`start_episode`)** | `0x14` | **571.75 px** | **300** | **0.28 px** |

**The committed Yoshi's Island 1 savestate restores into engine mode `$7E:0100 = 0x08` (file selector), not interactive gameplay `0x14`.** Any script that loads it
and immediately plans controls a non-interactive engine: WRAM still moves, so naive
sanity checks pass, but input handling differs. The preamble alone is worth
454.6 px of progress (3.4x) and a 13x change in sim-to-real alignment error. This
is why §10.6's numbers were refreshed, why the gameplay-mode poke that used to be
copy-pasted in ~15 scripts now lives in one helper
(`SnesLibretroEmulator.start_episode`), and why `src/environment/wram.py` documents
the mode byte.

#### 10.38.2 Reproduction Audit (`--reproduction-check`)

Re-executes the protocol `--repeats` times on separate seeds *without* overwriting
the published artifact, and reports each row against the committed value
(`results/mpc_reproduction_metrics.json`). On the corrected protocol: Hard PINN
420.9 ± 266.2 px (one seed falls in a pit at frame 177), Soft PINN 52.8 ± 13.3 px,
MLP 49.4 ± 14.6 px, random baseline 241.5 ± 0.0 px. Two statements follow: the
ordering of §10.6 is stable, and a single closed-loop row carries an uncertainty far
larger than the difference between the two black-box baselines - so §10.6 is read
as ranking evidence, and the multi-seed §8.4 machinery is what supports magnitude
claims.

#### 10.38.3 Manifest and Freshness Gates

`results/MANIFEST.md` indexes artifact → writer module → regenerating command →
README section, declares which checkpoints each closed-loop artifact loads, and is
enforced by `tests/test_results_manifest.py` (ownerless artifacts, fictional
writers, dangling claims, growing `_meta` exemptions, stale checkpoint inputs and
headline-number drift all fail CI). The freshness gate is what flagged §10.6;
the copy of the master table in §10.27 also carried a Dyna-PPO row that had been
pasted from the MPC row (164.75 px instead of the 115.00 px / 173 frames recorded in
`results/dyna_ppo_metrics.json`) and is now corrected.
The freshness gate compares git commit dates, so CI checks out with `fetch-depth: 0`
and a shallow clone skips that single gate instead of judging every pair a tie - the
first push of this batch failed precisely there, which is also why the workflow now
republishes the failing assertions as check annotations (readable without a GitHub
session, unlike Actions logs).

---
