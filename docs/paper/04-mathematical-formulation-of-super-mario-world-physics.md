[← 3](03-snes-hardware-architecture-wram-reverse-engineering.md) · [Contents](../../README.md#contents) · [5 →](05-evaluated-machine-learning-architectures.md)

## 4. Mathematical Formulation of Super Mario World Physics

### 4.1 Fixed-Point Arithmetic and Discrete Kinematic Consistency
In the 65816 assembly engine, player position is maintained as a 24-bit fixed-point accumulator comprising 16 bits of integer pixels and 8 bits of fractional subpixels. Each pixel is partitioned into 16 subpixels (where each increment in the subpixel's high nibble corresponds to $1/16$ of a pixel).

Consequently, the continuous real coordinate of the character at any frame $t$ is exactly:

$$X_t = X_{\text{pix}, t} + \frac{X_{\text{sub}, t}}{16.0 \times 16.0} \times 16.0 = X_{\text{pix}, t} + \frac{X_{\text{sub}, t}}{16.0}$$

$$Y_t = Y_{\text{pix}, t} + \frac{Y_{\text{sub}, t}}{16.0}$$

At every simulation step ($\Delta t = 1$ frame), the engine's kinematic routine accumulates the instantaneous velocity:

$$X_{t+1} = X_t + \frac{v_{x, t}}{16.0}$$

$$Y_{t+1} = Y_t + \frac{v_{y, t}}{16.0}$$

#### Discrete Kinematic Consistency Identity:
In the absence of instantaneous stage wraps or hard wall clammings, the displacement $\Delta X_t = X_{t+1} - X_t$ is strictly linear with respect to velocity $v_{x,t}$, governed by the invariant constant ratio:

$$\frac{\Delta X_t}{v_{x,t}} = \frac{1}{16.0} = 0.0625\quad [\text{pixels} \cdot \text{subpixel}^{-1}]$$

This equation represents an exact discrete numerical integration identity enforced by the game engine's computational routines (rather than a classical continuous conservation law in the Noetherian sense). Any forward model predicting $\hat X_{t+1} \ne X_t + \frac{v_{x, t}}{16.0}$ introduces a structural kinematic violation relative to the engine's arithmetic - with the velocity the frame *starts* with, which is the velocity §10.49 measured the console accumulating and §10.53 trained as a prediction target.

*Measured:* on the 6,329 training transitions of the gameplay recording (§10.54): exact on 93.82% of them, median residual 0.0000 px. Scored against $\hat v_{x,t+1}$ instead it is 17.90% and 0.0625 px, so a violation has to be defined with the frame's initial velocity - and on the 2,755 frames that carry no collision flag at all it is exact on 96.99% (§10.55 reads the same fact as a fitted integration coefficient).

Four implementations of the same sentence used to define the violation against $\hat v_{x,t+1}$ instead - `ResidualDynamics`, `ProjectedDynamics`, `HardResidualPINNDynamics` and `AnalyticalKinematicsDynamics` advanced position with the velocity the network predicts for the next frame by default, which §10.54 recorded as a documentation-and-code disagreement rather than as a second convention. All four now default to the carried velocity (`src.utils.kinematics.DEFAULT_POSITION_VELOCITY`); the next-frame form survives as `PUBLISHED_POSITION_VELOCITY`, which every pipeline that reproduces a committed artifact passes by name, because every artifact was recorded with it. §10.53 measures what that choice costs a trained model, and §10.55 that it is a second-order term the engine does not have.

### 4.2 Vertical Dynamics: Asymmetric Gravity and Jumping Mechanics
Vertical acceleration in *Super Mario World* displays an intentional input-modulated physical asymmetry:
1. **Standard Jump Impulse (B Button):** Pressing `B` injects an immediate negative vertical velocity:

   $$v_{y, 0} \in [-64, -80]\text{ subpixels/frame}$$
   (modulated by prior horizontal running momentum).
   *Measured: 7.39% of the recorded vertical velocities lie below -80, reaching -112.0 - this is the impulse the engine injects at take-off, not a bound it enforces, and a model that clamps to it rewrites real states (§10.54).*
2. **Spin Jump (A Button):** Injects lower initial vertical velocity ($v_{y,0} \approx -56$ subpixels/frame), granting invulnerability against certain hazard blocks.
3. **Ascent Gravity (Holding Jump Button):** While the jump button is held active during ascent ($v_y \lt 0$), effective gravity is reduced:

   $$g_{\text{held}} = +3.0\text{ subpixels/frame}^2 = +0.1875\text{ pixels/frame}^2$$
   *Measured on airborne-ascent frames: 86.51% of 1,386 step +3.0 and 7.58% step +6.0.*
4. **Descent Gravity (Released Button or Falling):** When the jump button is released early, or after the apex ($v_y \ge 0$), gravity doubles:

   $$g_{\text{fall}} = +6.0\text{ subpixels/frame}^2 = +0.3750\text{ pixels/frame}^2$$
   *Measured by stratum (§10.54): released descent is the only stratum where +6.0 is common (31.49% of 867 frames), while released *ascent* still steps +3.0 in 84.21% of 532 - the button-state gate is not separable in this recording. §10.52 collected the excitation that shows it and §10.56 identifies its coefficient.*
5. **Terminal Fall Velocity:** Downward velocity is documented as clamped in hardware:

   $$v_{y} \le v_{y, \text{term}} = +64.0\text{ subpixels/frame} = +4.0\text{ pixels/frame}$$
   *Measured: 13.37% of the recorded vertical velocities exceed +64, reaching 70.0, and 40.95% of released-descent frames show no vertical increment at all, which is where a clamp would show up if it were enforced here. It is a documented parameter this telemetry never shows the engine applying, so no model may treat +64 as an absolute bound on observed data.*

### 4.3 Horizontal Dynamics: Friction, Traction, and Skidding
Horizontal movement is governed by velocity saturation and discrete acceleration ramps. The three figures below are **speed classes** - the cap of each movement class - and not successive bounds on one variable, which is the distinction §10.49 had to re-establish after four sections scored ceiling estimates against the largest of them:
1. **Standard Walking:** $|v_x| \le 20\text{ subpixels/frame}$ ($1.25\text{ pixels/frame}$);
2. **Running (Holding Button Y/X):** $|v_x| \le 48\text{ subpixels/frame}$ ($3.0\text{ pixels/frame}$);
3. **Maximum Sprint (P-Meter Active):** $|v_x| \le 72\text{ subpixels/frame}$ ($4.5\text{ pixels/frame}$);
   *Measured:* on 6,329 transitions: 54.88% of frames exceed 20 and 0.0632% exceed 48, while nothing exceeds 72 ($\max |v_x| = 49.0$). These are three speed classes, not three successive bounds, and §10.49 is what re-scored the sections that read 72.0 as the engine's maximum.
4. **Friction and Skidding:**
   * Releasing directional input decelerates Mario gradually to zero through surface friction.
   * Inverting direction while running triggers the skidding state (*skid*), applying increased deceleration ($a_{\text{skid}} \approx 4\text{ to }6\text{ subpixels/frame}^2$).
   * Measured, this is the one §4 rule with no single coefficient: §10.56 identifies the horizontal law as 23 surviving terms whose run-button and coast contributions are spread over contact-and-direction interactions, and the bare coast term on $v_x$ does not survive thresholding at all.
5. **Ground Contact Gates the Gravity Step:** When $c_{t, \text{ground}} = 1$ (solid floor) and no jump action is commanded ($a_{t, \text{jump}} = 0$), the engine suppresses the gravity increment rather than stopping the body:

   $$v_{y, t+1} = v_{y, t} + g \cdot (1 - c_{t, \text{ground}})$$
   *Measured:* on the 2,943 grounded frames with no jump commanded the vertical increment is exactly zero on 90.72% of them, while the recorded $v_y$ is exactly zero on 0.00% (median $|v_y| = 6.0$, under all three ways of conditioning the stratum). The flag gates the gravity step; it does not stop the body. `GroundContactConsistencyLoss` and `AnalyticalKinematicsDynamics` both implement this rule by default; the retracted form is reached only as `PUBLISHED_CONTACT_RULE`, which the pipelines that reproduce a committed artifact pass by name.
6. **Ceiling Collision:** Striking a solid block from below with $v_y \lt 0$ immediately nullifies or inverts velocity to +1 to +8 subpixels/frame.

The `$7E:0077` collision byte that sets $c_{t, \text{ground}}$ is therefore a contact flag and not a rest flag (§10.54), which is why a model trained to zero the vertical velocity on every grounded frame is being asked to predict something the console does not do.

---
