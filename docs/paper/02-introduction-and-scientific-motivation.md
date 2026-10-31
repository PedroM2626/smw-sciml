[← 1](01-project-overview-abstract.md) · [Contents](../../README.md#contents) · [3 →](03-snes-hardware-architecture-wram-reverse-engineering.md)

## 2. Introduction and Scientific Motivation

At the nexus of Model-Based Reinforcement Learning (MBRL) and Physics-Informed Machine Learning (PIML), constructing accurate **World Models** capable of forward-simulating environmental state transitions $s_{t+1} = f(s_t, a_t)$ is essential for trajectory optimization and planning algorithms such as Model Predictive Control (MPC) and Monte Carlo Tree Search (MCTS).

Historically, the literature has addressed this challenge via two dominant paradigms with significant caveats:
1. **Pixel-Based Visual Models (Computer Vision / VAEs / World Models):** While general, they require heavy convolutional stacks, demand millions of environment interactions, incur high inference latency, and suffer from compounding visual degradation (*pixel blur* and *hallucination*).
2. **Black-Box State-Based Statistical Models (MLP / RNN / Transformers):** When fed numerical state vectors (coordinates, velocities), these networks treat physical state components as arbitrary statistical distributions. They fail to exploit fundamental invariances, such as the exact differential relation:

$$\frac{d\vec{x}}{dt} = \vec{v}$$

In platform games running on classic 16-bit consoles like the Super Nintendo Entertainment System (SNES), CPU physics routines are governed not by continuous stochastic differential equations, but by **deterministic hybrid discrete-time dynamical systems** operating at $60\text{ Hz}$ using fixed-point arithmetic.

### The Central Research Question:
> *"Does knowing the game physics help? If so, exactly how much does it help, and under which architectural formulation (soft penalty regularization in the loss vs. hard structural inductive bias) should this physics be embedded?"*

---
