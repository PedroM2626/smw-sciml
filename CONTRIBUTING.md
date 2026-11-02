# Contributing to smw-pinn

Short rules for keeping this repo reproducible. Code, comments, docstrings,
commit messages and docs are **English only** - the audience is the paper review,
and a mixed-language repository makes every search and quote ambiguous.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
# CPU-only fallback: ... --extra-index-url https://download.pytorch.org/whl/cpu
git lfs install && git lfs pull   # datasets, checkpoints, videos
```

You also need your own dump of the commercial ROM (SHA-1 in README);
never commit ROMs, savestates with personal data, or API keys.

## Dependency envelope

`torch>=2.5.1,<2.15`, `torchvision>=0.20.1,<0.30`, `numpy>=1.24.0,<2.6` are
validated on RTX 4070 + CUDA 12.6 (torch 2.14.1, the newest release with a CUDA build) and on
CPU with Python 3.13 + NumPy 2.5; the published artifacts themselves were recorded with torch
2.5.1 + CUDA 12.1. Install the CUDA build from the `cu126` index: the older `cu121` index stops at
torch 2.5.1, and with a wider cap pip would quietly take the CPU-only wheel from PyPI on Windows.
`scikit-learn<1.6` is a hard requirement, not a validation cap: gplearn 0.4.2, the latest
release, calls an estimator API that scikit-learn 1.6 removed. Dependabot is configured
(`.github/dependabot.yml`) to stay inside these caps — widen them only with a
full hardware re-validation (benchmarks + `make test-cov` on CUDA). CI uploads
`pytest.log` as an artifact on failure; check it before re-running blindly.

`requirements.txt`/`pyproject.toml` are install *ranges* (they must stay installable on
both the CUDA reference box and the CPU CI image). `requirements.lock` is a separate,
pinned `pip freeze` snapshot of the exact reference environment, kept as an audit record
of what produced the published numbers - regenerate it with `python -m pip freeze >
requirements.lock` after re-validating, and do not install it as-is on other hardware.

## Canonical commands (use these, not ad-hoc scripts)

```bash
make lint         # ruff check src tests scripts — must pass
make format       # ruff format (the same version CI and pre-commit enforce)
make format-check # ruff format --check — must pass
make typecheck    # mypy on the whole src/ package — must pass
make test         # fast unit tests
make test-cov     # tests + coverage gate (fail under 44%)
make check-all    # lint + format-check + typecheck + test-cov, i.e. the CI gate
make reproduce    # 2-epoch CPU smoke benchmark (configs/reproduce.yaml)
make smoke-all    # seconds-scale runs of the slow studies -> results_smoke/
make benchmark sample-efficiency multiseed
```

`make` is usually missing on Windows, where this project was developed: install
the package (`pip install -e ".[dev]"`) and use the identical console script
instead - `smw-pinn check-all`, `smw-pinn test-cov`, `smw-pinn smoke-all`, or
`smw-pinn run <module> [args...]` for any documented entry point (`src/cli.py`).
`smw-pinn list` prints the catalog.

Run entry points as modules (`python -m src.training.benchmark_experiment`),
never `cd src/` + relative imports. `src/` must not contain `sys.path` hacks;
only `scripts/` bootstraps the repo root.

## Code conventions

- **Logging, not `print()`:** `from src.utils.logging import get_logger; log = get_logger(__name__)`.
  `print()` is allowed only in `scripts/` (CLI tools). The two legitimate `src/` exceptions
  are the console runner `src/cli.py` and the JSON-stdout contract in
  `src/evaluation/analytical_baselines.py`; `tests/test_no_print_in_src.py` enforces this by
  AST and fails if a `print` appears anywhere else (add a *justified* allowlist entry there
  if a new console-stdout module is genuinely required).
- **Determinism:** seed via `src.utils.seed.set_global_seed`; DataLoaders take
  an explicit `seed=` (seeded `torch.Generator` + `seed_worker`). Never use
  unseeded `shuffle=True` or bare `np.random` in training/eval code.
- **Configs:** shared knobs live in `configs/*.yaml`; entry points accept
  `--config` with CLI-overrides-file semantics (`src/utils/config.py`).
  Warn (don't silently ignore) on unknown keys. A config key the parser does not
  know is silently ignored, so `tests/test_smoke_runs.py` checks each
  `configs/smoke_*.yaml` against the real `--help` output of its entry point.
- **Paths:** never hardcode `data/raw/...`, `src/environment/bin/...` or
  `results/...`. Resolve them through `src/utils/paths.py` (`require_rom()`,
  `results_file()`, `checkpoint_file()`, ...), which is repo-root anchored and
  honors the `SMW_ROM` / `SMW_CORE` / `SMW_DATA_DIR` overrides. Emulator addresses
  live in `src/environment/wram.py`, never as bare hex literals.
- **Metrics:** aggregate MSE hides scale imbalance — report per-variable
  metrics (`src/evaluation/per_variable_metrics.py`) alongside it, and
  multi-start rollouts instead of a single trajectory.
- **Tests:** every new module gets a test file. Emulator-dependent tests must
  use `@requires_emulator` from `tests/conftest.py` (Windows DLL doesn't load
  on Linux CI — tests must skip, never error).
- **Types:** `make typecheck` runs mypy over the whole `src/` package - the science
  library, the ctypes emulator wrapper, and every study and training entry point (147
  files). New or edited code anywhere under `src/` must pass it. Pre-commit (`pre-commit install`) runs `ruff check --fix`
  and `ruff format` with the same pinned version as CI.
- **Result artifacts:** a new `results/*.json` must be written through
  `src/utils.provenance.write_metrics` (so it carries `_meta`) and registered in
  `results/MANIFEST.md` with its writer module, regenerating command and README
  section. `tests/test_results_manifest.py` fails otherwise - it also refuses
  artifacts whose checkpoints were regenerated after the artifact was recorded.
- **Regression gate:** if you change training/eval code, check
  `tests/test_metrics_regression.py` still passes; regenerate `results/*.json`
  on GPU and update the tables that cite them in `docs/paper/` (§8.1–§8.4 and §10.x).
- **Pixel datasets are huge:** `scripts/record_pixel_gameplay.py` defaults to
  frame stride + downscale for a reason (full-res frames ≈ GBs). Keep frame
  `.npz` files in Git-LFS and never commit ROMs or smoke-test dirs.

## Pull requests

- `make check-all` green (CI runs lint, format-check, typecheck and `make test-cov`
  on ubuntu-latest).
- Small, focused PRs with a description of what changed and which
  `results/*.json` / README tables were affected, if any.
- Don't commit: `.venv/`, `runs/`, `*.mp4/*.gif` outside `results/figures/`,
  ROMs, `.coverage`, or smoke-test dirs (`results_smoke*/`).
