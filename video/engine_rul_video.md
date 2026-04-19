---
noteId: "a46ea3203bf011f1a0d1c39625346603"
tags: []
marp: true
theme: "default"
size: "16:9"
paginate: false
backgroundColor: "#0e1117"
color: "#e8e8e8"
style: "section { font-family: 'Inter', 'Segoe UI', sans-serif; padding: 60px 80px; } h1 { color: #ffffff; font-size: 64px; margin-bottom: 0.2em; } h2 { color: #f5a623; font-size: 40px; margin-bottom: 0.4em; } .metric { color: #4fc3f7; font-weight: 700; font-size: 72px; } .caption { color: #a0a0a0; font-size: 22px; } img { border-radius: 8px; }"
---

<!-- _class: lead -->

# Engine Failure & Remaining Useful Life Prediction

## NASA C-MAPSS · CNN vs Physics-Informed Neural Network

<br>

**Final Project — RCSSTEAP, Beihang University**

Kodie Amo Kwame (LS2525226)
Sumara Alfred Salifu (LS2525245)
Peta Mimi Precious (LS2525255)

<!-- note: Jet engines fail. When they do, lives and millions of dollars are at stake. This is our approach to predicting engine failure before it happens — physics-informed deep learning on NASA's turbofan dataset. -->

---

## The Dataset — NASA C-MAPSS

![bg right:55% fit](../output/sensor_trajectories.png)

- 4 fault modes: **FD001 – FD004**
- 21 sensors, 3 operating settings
- Engines run to failure
- **FD001**: 100 train / 100 test engines

<!-- note: NASA's C-MAPSS dataset simulates turbofan engine degradation across four fault modes — F D zero zero one through F D zero zero four. Each engine runs until failure, logging twenty one sensors every cycle. We focus on F D zero zero one — one hundred training engines, one hundred test engines. -->

---

## What Is RUL?

![bg right:55% fit](../output/rul_trajectory_example.png)

**Remaining Useful Life** — cycles left before failure.

- Early life: RUL is effectively **constant**
- Degradation becomes observable only in the **final phase**
- We use a **piecewise-linear cap at 125 cycles**

<!-- note: Remaining Useful Life, or R U L, is the number of cycles an engine has left before failure. Early on, R U L is effectively constant — degradation only becomes observable in the final phase. We cap R U L at one hundred twenty five cycles so the model focuses on the degradation regime. -->

---

## Two Models

![bg right:50% fit](../output/network_architectures.png)

**1D-CNN (baseline)**
Conv1D(F=16, k=5) → GAP → Dense(32) → Dense(1)

**PINN (physics-informed MLP)**
Dense(96) → Dense(48) → Dense(1)
**+ physics loss**

<!-- note: We built two models. A baseline one-D convolutional network that learns purely from data — Conv1D, global average pooling, two dense layers. And a physics-informed neural network — a P I N N — that knows R U L must decrease monotonically by exactly one cycle each step. -->

---

## The Physics Loss

The PINN adds two terms on adjacent-cycle pairs:

**Monotonicity:** `ReLU(RUL(t+1) − RUL(t))²`
> RUL should never increase.

**Drift:** `(RUL(t) − RUL(t+1) − 1)²`
> RUL should drop by exactly one per cycle.

**λ_mono = 0.15 · λ_drift = 0.05**

<!-- note: The P I N N adds two physics terms. A monotonicity penalty that discourages predicting R U L going up — because a degraded engine cannot un-degrade. And a drift penalty that enforces a one-cycle decrease between consecutive windows. Lambda mono zero point one five, lambda drift zero point zero five. -->

---

## Training

![bg right:55% fit](../output/training_curves.png)

- **14,207** training windows, **3,524** validation
- CNN: 18 epochs · PINN: 22 epochs
- **Total wall time: 20.5 s**
- PINN converges to lower val loss thanks to the physics prior

<!-- note: Both models trained in under twenty one seconds on fourteen thousand two hundred seven windows. The P I N N converges to lower validation loss thanks to the physics regularizer. -->

---

## Predictions — CNN vs PINN

![bg left:50% fit](../output/pred_vs_actual_cnn.png)
![bg right:50% fit](../output/pred_vs_actual_pinn.png)

<br><br><br><br><br>

**Left:** CNN — scattered
**Right:** PINN — hugs the diagonal, especially near end of life

<!-- note: On one hundred unseen test engines, the C N N predictions scatter widely. The P I N N predictions hug the diagonal far more tightly — especially near end of life, where early failure is most costly. -->

---

## Results

![bg right:45% fit](../output/model_comparison.png)

|  | CNN | PINN | Δ |
|---|---:|---:|---:|
| **RMSE** | 18.68 | **14.64** | **−21.6%** |
| **MAE** | 13.69 | **11.14** | **−18.6%** |
| **NASA score** | 1107.5 | **338.6** | **−69.4%** |

<!-- note: The numbers: C N N R M S E eighteen point six eight, P I N N fourteen point six four — a twenty one point six percent improvement. On N A S A's asymmetric score function, which punishes late predictions, we drop from eleven hundred seven to three hundred thirty eight — a sixty nine percent reduction. -->

---

## Error Profile

![bg left:50% fit](../output/error_histogram.png)
![bg right:50% fit](../output/per_engine_rul.png)

<br><br><br><br><br>

**Left:** Error distribution — tighter for PINN
**Right:** Per-engine RUL trajectories track ground truth smoothly

<!-- note: Errors are tighter, and per-engine R U L trajectories track the truth line smoothly. No post-hoc smoothing — this comes from the physics prior alone. -->

---

<!-- _class: lead -->

# Physics-Informed Learning
# = Safer, More Accurate RUL

<br>

**Thank you**

Kodie Amo Kwame · Sumara Alfred Salifu · Peta Mimi Precious
RCSSTEAP, Beihang University

<!-- note: Physics-informed learning turned a small network into a safer, more accurate R U L estimator. Thank you. -->
