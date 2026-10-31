[← 10.7](10-07-amortized-policy-optimization-dyna-ppo-zero-shot-model-to-real-transfer.md) · [Contents](../../README.md#contents) · [10.9 →](10-09-canonical-model-free-ppo-baseline-vs-pinn-mbrl-sample.md)

### 10.8 Dynamic Hazard Perception (WRAM Sprites & Rex Evasion)

To solve the terminal boundary identified in Section 10.7.3 (where Mario collided with the first stage enemy, *Rex*, at $X \approx 131$), we reverse-engineered the SNES Working RAM sprite tables:
- **Sprite Status:** `$7E:14C8` to `$7E:14D3` (12 slots; status $\ge 8$ denotes active interactive execution).
- **Sprite Classification:** `$7E:009E` to `$7E:00A9` (ID `0x05` = Rex, `0x0F` = Goomba, etc.).
- **Sprite World Coordinates:** High/Low 16-bit registers `$7E:14E0 / $7E:00E4` ($X_{\text{sprite}}$) and `$7E:14D4 / $7E:00D8` ($Y_{\text{sprite}}$).
- **Sprite Velocities:** Signed 8-bit registers `$7E:00B6` ($v_{x,\text{sprite}}$) and `$7E:00AA` ($v_{y,\text{sprite}}$).

We formulated a **12-Dimensional Extended State Representation**:

$$s_{\text{ext}} = [s_{\text{mario}} \in \mathbb{R}^8, \Delta X_{\text{hazard}}, \Delta Y_{\text{hazard}}, v_{x,\text{hazard}}, \text{hazard flag}]$$
where $\Delta X_{\text{hazard}} = X_{\text{sprite}} - X_{\text{mario}}$ provides egocentric hazard telemetry.

#### Hardware Evasion Benchmark on Physical SNES Console:
A critical architectural discovery emerged regarding SNES joypad polling: register `$7E:0016` (`Controller_Press`) requires an **active low-to-high trigger edge** to execute a jump impulse ($v_y = -72$). If button `B` is held down continuously across landing, the engine registers it as merely held, aborting subsequent jump impulses. By monitoring $\Delta X_{\text{hazard}}$, the agent releases `B` during descent to prime the trigger edge, then initiates a sustained high-arc leap ($v_y = -73$) upon touching ground:

| Agent Controller | Sensory Perception | Real Console Progress ($\Delta X$) | Real Survival (Frames) | Rex Encounter Outcome |
| :--- | :--- | :---: | :---: | :--- |
| **Blind Agent (Dyna-PPO)** | 8D Kinematics (No Sprites) | +115.0 px | 173 frames | Fatal impact at $X \approx 131$ |
| **Sprite-Aware Agent** | **12D WRAM Sprite Telemetry** | **+328.9 px (2.86x higher)** | **500 / 500 frames (100%)** | **Clean jump over Rex apex to $X \gt 344$** |

![Sprite Evasion Trajectories](../../results/figures/sprite_evasion_trajectories.png)

---
