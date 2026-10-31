[← 10.19](10-19-autonomous-multi-entity-mpc-planning-on-real-snes.md) · [Contents](../../README.md#contents) · [10.21 →](10-21-empirical-grounding-of-tilemap-pinn-wram-7e-c800.md)

### 10.20 Spatial Discrete Tilemap Perception via WRAM (`$7E:C800`) & Tilemap-PINN

To address the blindness of coordinate-only dynamics to static environmental geometry (pipes, ledges, blocks), we reverse-engineered the SNES Working RAM level block buffer:
- **WRAM Memory Mapping:** In horizontal SMW levels, 16x16 pixel blocks are indexed sequentially across 32 subscreens in `$7E:C800`:

  $$\text{addr} = 0\text{xC800} + \left(\lfloor X / 256 \rfloor \times 0\text{x01B0}\right) + \left(\lfloor Y / 16 \rfloor \times 16\right) + \left(\lfloor X \bmod 256 / 16 \rfloor\right)$$
- **Local Spatial Patch Extraction:** `snes_emulator.py` implements `get_local_tilemap_patch(mario_x, mario_y, radius=3)` returning a discrete $7 \times 7$ grid of surrounding blocks categorized into `0: Air`, `1: Solid Terrain`, `2: Hazard`, `3: Slope`.
- **Hybrid Tilemap-PINN Architecture (`src/models/tilemap_pinn.py`):**
  - Convolutional Tile Encoder: `Embedding(4, 8) -> Conv2d(8, 16) -> Conv2d(16, 24) -> AdaptivePool -> LayerNorm` (32 terrain features).
  - Residual Contact Head: Predicts anticipatory contact flags (`c_ground`, `c_left`, `c_right`) conditioned on geometry *before* impact occurs.
  - Kinematic Guarantee: Preserves strict 0.0% kinematic violation ($\Delta X = v_x/16.0$) via hard structural integration layers.
  - Test Suite: 100% passing tests in `tests/test_tilemap.py`.

---
