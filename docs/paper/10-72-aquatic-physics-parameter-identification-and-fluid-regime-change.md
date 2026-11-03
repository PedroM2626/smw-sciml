[← 10.71](10-71-physics-informed-reinforcement-learning-across-every-stage.md) · [Contents](../../README.md#contents) · [10.73 →](10-73-end-to-end-pixel-perception-with-physics-informed-reinforcement-learning.md)

### 10.72 Aquatic Physics Parameter Identification and Fluid-Regime Transition

10.71 established that closed-loop Physics-Informed Stage PPO (`PIML-PPO`) clears every terrestrial stage the world map hands over (`0x2A`, `0x29`, `0x27`). Finishing `0x27` on the console moves the overworld cursor from node `(152, 104)` to `(184, 72)` and unlocks **`0x26` (*Yoshi's Island 4*)**, saved as `data/raw/smw_map_round_2_184_72_t26.state` beside the post-clear world map `data/raw/smw_world_map_after_round_1_152_104_t27.state`.

In `0x26`, Mario starts on an elevated wooden pier at `x = 16.0 px, y = 272.6 px` over a lake. Moving past `x = 304 px` drops the body into the water, triggering the engine's internal fluid routines via the in-water WRAM flag `$7E:0075 = 1`. This presents an inverse system-identification problem: **how does the discrete dynamical law change across the phase boundary between air and water, and can an inverse solver recover the fluid parameters to prevent catastrophic forward prediction drift?** `src/inverse/aquatic_physics_identification.py` measures both regimes on authentic hardware.

#### 10.72.1 Physical Formulation of the Aquatic Dynamical Regime

In terrestrial air (`$7E:0075 = 0`), vertical dynamics follow the asymmetric gravitational acceleration identified in 10.40:

$$\Delta v_{y,t} = \begin{cases} g_{\text{hold}} \approx 3.0 \text{ subpx/frame}, & \text{if } B \text{ is held and } v_{y,t} \lt 0 \\ g_{\text{fall}} \approx 6.0 \text{ subpx/frame}, & \text{otherwise} \end{cases}$$

subject to a jump impulse $J_{\text{air}} \in [-80, -96] \text{ subpx/frame}$ and terminal velocity $v_{y,\text{term}} = 64.0 \text{ subpx/frame}$.

Upon entering the fluid medium (`$7E:0075 = 1`):
1. **Buoyant Gravitational Damping:** The downward acceleration on sinking frames drops from $\approx 6.0 \text{ subpx/frame}$ to an effective buoyant gravity $g_{\text{water}} \approx 0.48 \text{ subpx/frame}$ — an **effective gravity reduction of over 91.8%** ($\frac{g_{\text{water}}}{g_{\text{air}}} = 0.0818$).
2. **Swim Stroke Impulse:** Pressing the jump button ($B$) does not trigger a terrestrial jump arc; instead, it executes a periodic swim stroke that resets vertical velocity to an upward impulse $J_{\text{water}} = -24.0 \text{ subpx/frame}$ (mean $-16.37 \text{ subpx/frame}$ across stroke durations), corresponding to roughly $29.3\%$ of the terrestrial jump impulse.
3. **Viscous Speed Ceiling:** Drag limits maximum horizontal swimming speed to $16.0 \text{ subpx/frame}$ (versus $37.0 \text{ subpx/frame}$ in air, a $56.8\%$ suppression).
4. **Medium Invariance of the Discrete Kinematic Identity:** Across all $1,430$ aquatic transitions in the evaluation split, the discrete kinematic position integration identity:

$$16 (\hat x_{t+1} - x_t) - v_{x,t} = 0$$

holds with a maximum residual of exactly **$0.0000$** and a fitted scale factor of **$16.0000$**. This confirms empirically that the $\frac{1}{16}$ subpixel divisor is an invariant property of the CPU accumulator bus, entirely decoupled from environmental drag or gravity.

#### 10.72.2 Empirical Identification and Forward Transfer Results

Across $4,000$ transitions recorded on real hardware ($2,570$ terrestrial air transitions and $1,430$ aquatic fluid transitions):

| Parameter / Physical Metric | Terrestrial Air (`$7E:0075 = 0`) | Aquatic Fluid (`$7E:0075 = 1`) | Ratio (Water / Air) |
|---|:---:|:---:|:---:|
| Effective Gravitational Acceleration | 5.873 ± 0.864 subpx/f | **0.480 ± 0.854 subpx/f** | **0.0818 (91.8% damping)** |
| Peak Upward Impulse ($\text{subpx}$) | -82.0 subpx | **-24.0 subpx** | 0.2927 |
| Maximum Horizontal Velocity ($\text{subpx/f}$) | 37.0 subpx/f | **16.0 subpx/f** | 0.4324 |
| Kinematic Subpixel Integration Scale | 17.04 (due to horizontal collisions) | **16.0000 (exact)** | 1.0000 |
| Max Kinematic Residual ($|16\Delta x - v_x|$) | 20.0 (wall contact frames) | **0.0000 (contact-free fluid)** | — |

Evaluating forward trajectory prediction error on the $1,430$ fluid transitions demonstrates the necessity of regime adaptation:

| Evaluated Forward Dynamics Model | Y Position MAE (px) | Y Position MSE ($\text{px}^2$) | Y Velocity MAE (subpx) | MSE Error Reduction |
|---|:---:|:---:|:---:|:---:|
| **Zero-Shot Terrestrial Model** ($g=5.87, J=-82.0$) | 1.5310 px | 6.0709 $\text{px}^2$ | 23.8094 subpx | Baseline null |
| **Adapted Aquatic Model** ($g=0.48, J=-24.0$) | **0.3298 px** | **0.5070 $\text{px}^2$** | **5.0159 subpx** | **12.0× lower error** |

Applying the terrestrial world model zero-shot to water transitions incurs severe vertical prediction drift ($1.53\text{ px}$ MAE, $6.07\text{ px}^2$ MSE) because it assumes Mario plunges downward at $6.0 \text{ subpx/frame}$ and overshoots swim strokes by a factor of three. Adapting the two free parameters ($g_{\text{water}}, J_{\text{water}}$) recovers true fluid motion with a **$12.0\times$ reduction in mean squared position error**, verifying that physics-informed world models adapt to novel fluid media with minimal transition data.

Regenerate: `python -m src.inverse.aquatic_physics_identification`. Artifact: `results/aquatic_physics_identification_metrics.json`. Savestates: `data/raw/smw_map_round_2_184_72_t26.state` and `data/raw/smw_world_map_after_round_1_152_104_t27.state`.
