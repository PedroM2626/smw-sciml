[← 6](06-loss-function-formulation-and-the-soft-pinn.md) · [Contents](../../README.md#contents) · [8 →](08-empirical-results-and-comparative-benchmark-tables.md)

## 7. Experimental Protocol, Data Partitioning & Hyperparameters

### 7.1 Data Integrity and Acquisition
1. All synthetic or fabricated trajectories were strictly rejected.
2. The official US retail ROM of *Super Mario World* was executed on stage *Yoshi's Island 1*.
3. The emulator ran in **Interactive In-Level Gameplay Mode `$7E:0100 = 0x14`**, ensuring real-time physics simulation driven by active controller inputs.
4. Recorded controller patterns included walking, sprinting with button `Y`, short hops and high jumps with button `B`, direction reversals (*skidding*), pipe collisions, and full resting stops.
5. Exactly **8,077 consecutive transitions** ($s_t, a_t, s_{t+1}$) were recorded into `data/raw/smw_gameplay_dataset.npz`.

### 7.2 Temporal Data Partitioning
To prevent data leakage across time-series, partitions were constructed via contiguous temporal episodes:
- **Training Set:** 6,329 transitions (78.4%);
- **Validation Set:** 392 transitions (4.8%);
- **Independent Test Set:** 1,356 transitions (16.8%).

### 7.3 Unified Hyperparameters
- **Optimizer:** AdamW ($\eta = 10^{-3}$, weight decay $\lambda = 10^{-5}$);
- **Batch Size:** 64 transitions;
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor $= 0.5$, patience $= 3$ epochs);
- **Early Stopping:** Monitored on $\mathcal L_{\text{val}}$ (patience $= 8$ epochs);
- **Maximum Epochs:** 35 epochs;
- **Fixed Seeds:** `torch.manual_seed(42)`, `np.random.seed(42)`.

---
