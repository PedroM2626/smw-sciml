# smw-pinn

**Physics-Informed Neural Networks vs. Statistical Models in Super Mario World** -
discrete dynamics modeling without computer vision, from genuine WRAM telemetry of a
real SNES, with closed-loop control flown on the console itself.

This is the landing page of the documentation site. The research document lives in the
[Full README](full-readme.md), rendered here through MathJax so every formula typesets -
GitHub stops rendering math after roughly its first 1,368 expressions on a page, which
the README's own Section 12 measured, and this site exists so the tail of that document
is readable. The canonical repository, with the issues, the releases and the artifacts
behind every number, is [PedroM2626/smw-sciml](https://github.com/PedroM2626/smw-sciml).

## The headline result

Four architectures, one benchmark, genuine frame-by-frame transitions recorded from
Working RAM during interactive gameplay (README Section 8):

| Architecture | Paradigm | Test loss (MSE) | Kinematic residual |
| :-- | :-- | :--: | :--: |
| Statistical MLP | black-box baseline | 16.4717 | 17,561.24 |
| Statistical LSTM | recurrent baseline | 39.2194 | 37,361.16 |
| Soft-Constrained PINN | loss-penalty regularization | 53.8167 | 108,520.99 |
| **Hard Residual PINN** | **structural inductive bias** | **0.5783** | **0.0019** |

Embedding the discrete kinematic identity into the computational graph beats penalizing
it in the loss by two orders of magnitude - and the sections after that benchmark
measure, among other things, where even that advantage is tolerance-bound, which
hand-written engine rules beat every learned model, and which conclusions had to be
corrected along the way.

## Quickstart

```bash
git lfs install          # datasets, checkpoints, figures and savestates are LFS objects
git lfs pull
python -m venv .venv
.venv\Scripts\activate   # Windows: the bundled Libretro core is a Windows DLL
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e `.[dev]`
make test                # or: smw-pinn test-cov
make reproduce           # the CPU benchmark of Section 8 in minutes
```

The commercial ROM is not distributed: supply your own dump and verify the SHA-1 the
README publishes (Section 11.2). Everything emulator-free - the identification studies,
the operator benchmarks, the conformal intervals of Section 10.56.1 - runs without it.

## Where things are

- [Full README (MathJax, no budget)](full-readme.md) - the research document.
- [results/MANIFEST.md](https://github.com/PedroM2626/smw-sciml/blob/master/results/MANIFEST.md) -
  every published number, indexed artifact-first, enforced by CI.
- [CHANGELOG.md](https://github.com/PedroM2626/smw-sciml/blob/master/CHANGELOG.md) -
  including the audited self-corrections the repository reports instead of quietly
  applying.
- [CITATION.cff](https://github.com/PedroM2626/smw-sciml/blob/master/CITATION.cff) -
  how to cite this software.

## License

MIT - see the repository. The Snes9x Libretro core and the commercial ROM follow their
own terms; neither is redistributed by the docs site.
