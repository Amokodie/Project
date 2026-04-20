# NASA C-MAPSS FD001 - CNN vs PINN for Remaining Useful Life

**Run wall time:** 20.5s  ·  **Subset:** FD001  ·  **Test engines:** 100  ·  **Feature channels:** 18 (dropped 6 constant sensors)

## 1. Headline result

| Model | Architecture | Test RMSE down | Test MAE down | NASA score down |
|---|---|---:|---:|---:|
| **1D-CNN** (data-only) | Conv1D(F=16,k=5) -> ReLU -> GAP -> Dense(32) -> Dense(1) | 18.68 | 13.69 | 1107.5 |
| **PINN** (data + physics) | Dense(96) -> Dense(48) -> Dense(1) | **14.64** | **11.14** | **338.6** |
| **PINN vs CNN** | | **-21.6%** | **-18.6%** | **-69.4%** |

> **Story:** Adding two physics penalties (monotonic decay + per-cycle drift) to a small MLP cuts RMSE by **22%**, MAE by **19%**, and the **NASA late-warning score by 69%** vs. a pure 1D-CNN on the same windows. Same wall-clock budget. See `model_comparison.png`.

## 2. Setup

- Window length: 30 cycles  ·  RUL cap: 125 cycles
- Train/val: 80/20 by **engine ID** (14207 / 3524 windows)
- Test: 100 held-out engines, last available window per engine
- Optimizer: Adam (lr 1e-3), batch 128, He init, seed 42
- PINN total loss = MSE + 0.15*L_mono + 0.05*L_drift on adjacent-cycle pairs

Architecture diagram: `network_architectures.png`.

## 3. Training dynamics - `training_curves.png`

Both models converge cleanly in <25 epochs. PINN reaches the CNN's final validation RMSE within ~4 epochs and keeps improving. Physics term is one to two orders of magnitude smaller than the data MSE - it acts as a **shape prior**, not the dominant loss.

## 4. Pred-vs-actual scatter

`pred_vs_actual_cnn.png` shows the CNN's right-side ceiling: many true-RUL = 100-125 engines pulled to ~110 with a long tail of severe under-predictions. `pred_vs_actual_pinn.png` tightens the cluster around y=x and removes the worst outliers.

## 5. Error distribution - `error_histogram.png`

Both bias and spread improve. The disappearance of the +30 to +60 cycle tail is the single biggest driver of the NASA-score gap, because that score punishes late predictions exponentially.

## 6. Per-engine sorted view - `per_engine_rul.png`

For near-failure engines (left side, RUL < 30) both models track tightly - these are the operationally critical predictions. The CNN spikes upward on several mid-life engines; the PINN's monotonicity penalty makes such jumps explicitly costly during training.

## 7. Single-engine cycle-by-cycle - `rul_trajectory_example.png`

Both models stay near the 125-cycle cap while the engine is healthy, then descend together once degradation appears in the sensors. PINN tracks the true slope better near end-of-life. In production the per-cycle jitter would be smoothed with a small EMA.

## 8. What the CNN learned - `cnn_filters.png`

Several of the 16 learned 1D filters specialise in single sensor rows (the high-signal channels: T30, T50, P30, Nf - the standard turbofan degradation indicators). Others learned mixed-sensor temporal patterns. This is why CNNs are competitive with no physics input: the convolution rediscovers "sensor X drifts with sensor Y" by gradient descent.

Sensor context: `sensor_trajectories.png` (raw FD001 traces, longest-running engine).

## 9. Why this matters

1. **Same wall-time, big quality gap.** Both finish in seconds on CPU.
2. **Physics != replacement for data.** Data MSE still does >90% of the lifting. Physics rules out unphysical predictions.
3. **The metric you optimise matters.** RMSE drops 22%; the operational NASA score drops 69%. PINN wins the metric operators care about.
4. **Interpretability angle.** Physics penalties are auditable in plain English ("RUL must not increase, must drop ~1/cycle"). A CNN cannot match that for regulators.
5. **Limit:** FD001 is a single-regime subset. Numbers will degrade on FD002/FD004 (6 regimes, 2 fault modes).

## 10. Suggested slide order

| # | Slide | Figure |
|---|---|---|
| 1 | Title + dataset framing | - |
| 2 | Two-model architecture summary | `network_architectures.png` |
| 3 | Headline metric comparison | `model_comparison.png` |
| 4 | Training dynamics | `training_curves.png` |
| 5 | Pred-vs-actual side-by-side | `pred_vs_actual_cnn.png`, `pred_vs_actual_pinn.png` |
| 6 | Error distribution | `error_histogram.png` |
| 7 | Per-engine sorted + one-engine trajectory | `per_engine_rul.png`, `rul_trajectory_example.png` |
| 8 | What the CNN learned | `cnn_filters.png` |
| 9 | Why PINN wins the operational score | discussion only |
| 10 | Limits & next steps (FD002/FD004) | - |
