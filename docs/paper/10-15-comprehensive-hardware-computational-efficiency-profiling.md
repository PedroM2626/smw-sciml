[← 10.14](10-14-safe-model-based-reinforcement-learning-deep-ensemble-safe.md) · [Contents](../../README.md#contents) · [10.16 →](10-16-out-of-distribution-ood-zero-shot-cross-stage-generalization-stage-a.md)

### 10.15 Comprehensive Hardware & Computational Efficiency Profiling

To assess the engineering feasibility of deploying physics-informed world models on real-time embedded hardware, we developed a formal profiling suite (`src/evaluation/benchmark_computational_efficiency.py`) executed on an NVIDIA GeForce RTX 4070 Laptop GPU:

| Model Architecture | Trainable Parameters | Theoretical FLOPs | CPU Single-Core Latency | CUDA Latency (Batch 1) | CUDA Throughput (Batch 256) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Statistical MLP** | 36,744 | 72,704 | 76.3 µs | 162.2 µs | 1,349,125 FPS |
| **Statistical LSTM** | 223,368 | 325,632 | 213.9 µs | 188.1 µs | 219,827 FPS |
| **Soft PINN** | 36,744 | 72,704 | 78.1 µs | 179.5 µs | 1,325,302 FPS |
| **Hard Residual PINN** | **36,486** | **72,192** | **138.4 µs** | **325.4 µs** | **675,682 FPS** |
| **Multi-Entity PINN (12D)** | 37,192 | 73,472 | 274.4 µs | 712.9 µs | 323,660 FPS |
| **Deep Ensemble (E=5)** | 182,430 | 360,960 | 686.9 µs | 1,703.6 µs | 141,430 FPS |

#### Profiling Key Takeaways:
- The **Hard Residual PINN** achieves over **675,000 forward transitions per second** in batched execution on consumer hardware, enabling 100,000-sample policy rollouts in less than 0.15 seconds.
- Single-instance inference takes **138.4 microseconds on a single CPU core**, operating at **7,225 FPS**—more than **120 times faster** than the real-time 60 Hz frame rate of the SNES console.

![Computational Efficiency Comparison](../../results/figures/computational_efficiency_comparison.png)

---
