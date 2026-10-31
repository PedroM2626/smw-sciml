[← 10.5](10-05-implications-for-model-based-reinforcement-learning-mbrl.md) · [Contents](../../README.md#contents) · [10.7 →](10-07-amortized-policy-optimization-dyna-ppo-zero-shot-model-to-real-transfer.md)

### 10.6 Closed-Loop Model-Based RL (MBRL) via Model Predictive Control (MPC)

To evaluate whether the learned physics models can serve as viable **World Models** for autonomous decision-making in reinforcement learning, we implemented a closed-loop **Model Predictive Control (MPC)** trajectory optimizer using the Cross-Entropy Method (CEM: $H=15$ frames horizon, $N=256$ candidate action sequences per step, 3 CEM refinement iterations).

The agent navigates stage *Yoshi's Island 1* directly inside the real headless SNES emulator for 300 simulation frames (5.0 seconds at 60 Hz), with the objective of maximizing horizontal progress $\Delta X$ while avoiding pit falls.

#### Closed-Loop Hardware Execution Results (Real SNES Console):

> [!IMPORTANT]
> **Refreshed 2026-09-22.** This table was originally recorded by a harness that
> restored the savestate without entering interactive gameplay mode. The committed
> Yoshi's Island 1 savestate comes up in engine mode `$7E:0100 = 0x08` (file
> selector), so those rows planned against a non-interactive engine. Every episode
> now starts through `SnesLibretroEmulator.start_episode()` (restore → force mode
> `0x14` → warm-up frames) and the MPC trials are seeded. Section 10.38 quantifies
> the effect of that preamble: 3.4x more progress and a 13x smaller alignment error
> for the same checkpoint. The previous row values are preserved in git history and
> in `results/mpc_preamble_probe_metrics.json`.

| World Model Controller | Total Progress ($\Delta X$) | Planning Alignment Error ($\|\hat s_{\text{pred}} - s_{\text{real}}\|$) | Mean Forward Velocity ($v_x$) | Survival (Frames) |
| :--- | :---: | :---: | :---: | :---: |
| **Hard PINN World Model** | **+571.8 px** | **0.28 px** | **+30.8 subpixels/frame** | **300 / 300 (100%)** |
| **Statistical MLP World Model** | +66.1 px | 90.26 px (**320x higher**) | +3.6 subpixels/frame | 300 / 300 (100%) |
| **Soft-PINN World Model** | +38.7 px | 219.25 px (**779x higher**) | +2.1 subpixels/frame | 300 / 300 (100%) |
| **Random Exploration Baseline** | +241.5 px | N/A | +12.9 subpixels/frame | 300 / 300 (100%) |

#### Key MBRL Takeaways:
1. **Perceptual Alignment with Reality:** The Hard PINN World Model achieved a mean one-step spatial alignment error of only **0.28 pixels**, compared to **90.26 pixels** for the Statistical MLP and **219.25 pixels** for the Soft PINN. Because the Hard PINN embeds discrete kinematic consistency by construction, its predicted trajectories adhere to the game engine's position integration rules.
2. **Propulsion and Forward Progress:** Guided by the Hard PINN, the MPC agent traversed **+571.8 pixels** with an average horizontal velocity of **+30.8 subpixels/frame** (i.e. sustained running, close to the 48 subpixel run tier), executing coordinated runs and jumps that translate directly into hardware advancement.
3. **Black-Box Models Are Worse Than Doing Nothing:** Under the corrected protocol the ranking is unambiguous: Hard PINN **571.8 px** > random actions **241.5 px** > MLP **66.1 px** > Soft PINN **38.7 px**. Both unconstrained planners spend their horizon on transitions the console never executes (90-219 px of one-step imagination error), which produces *hesitation* - the agent stands still while a random walk moves forward. This is optimism under hallucinated dynamics, now measurable instead of inferred.
4. **Uncertainty of a Single Closed-Loop Row:** `results/mpc_reproduction_metrics.json` re-runs this exact protocol with three seeds: the Hard PINN row is 420.9 ± 266.2 px (one seed dies in a pit at frame 177 while two clear 300 frames), the MLP/Soft rows are ± 14 px, and the random baseline reproduces exactly (it is seeded internally). Closed-loop magnitudes should therefore be read as *ordering* evidence, not as precise point estimates.

![MBRL Closed-Loop Trajectories](../../results/figures/mbrl_mpc_trajectories.png)

---
