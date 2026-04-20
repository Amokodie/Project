---
noteId: "a46ea3203bf011f1a0d1c39625346603"
tags: []
marp: true
theme: "default"
size: "16:9"
paginate: true
backgroundColor: "#0e1117"
color: "#e8e8e8"
style: "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap'); section { background: radial-gradient(ellipse at 30% 15%, #1f3a68 0%, #0b1220 55%, #142034 100%); font-family: 'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', sans-serif; padding: 52px 72px 60px; color: #e6eaf2; position: relative; } section::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 5px; background: linear-gradient(90deg, transparent 0%, #0ea5e9 25%, #38bdf8 50%, #22d3ee 75%, transparent 100%); } section.lead { text-align: center; background: radial-gradient(circle at 50% 30%, #1f3a68 0%, #0b1220 70%); } .eyebrow { display: inline-block; font-size: 18px; letter-spacing: 3.2px; text-transform: uppercase; color: #7dd3fc; font-weight: 600; margin-bottom: 20px; } h1 { color: #ffffff; font-size: 58px; font-weight: 800; letter-spacing: -1.4px; margin: 0 0 12px 0; line-height: 1.05; } h2 { color: #ffb347; font-size: 34px; font-weight: 700; border-left: 6px solid #ffb347; padding-left: 18px; margin: 0 0 18px 0; letter-spacing: -0.3px; } h3 { color: #7dd3fc; font-size: 24px; font-weight: 600; margin: 8px 0 4px 0; letter-spacing: 0.2px; } p, li { font-size: 23px; line-height: 1.5; color: #e6eaf2; } strong { color: #ffd166; font-weight: 600; } em { color: #a5b4fc; font-style: italic; } table { font-size: 22px; border-collapse: collapse; width: 100%; margin-top: 8px; } th { color: #7dd3fc; font-weight: 700; border-bottom: 2px solid rgba(125,211,252,0.28); padding: 10px 14px; text-align: left; font-size: 18px; text-transform: uppercase; letter-spacing: 1px; } td { padding: 10px 14px; border-bottom: 1px solid rgba(125,211,252,0.12); font-family: 'JetBrains Mono', 'Consolas', monospace; } td:first-child { font-family: 'IBM Plex Sans', sans-serif; } code { background: #0a1426; color: #fcd34d; padding: 2px 10px; border-radius: 6px; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 20px; border: 1px solid rgba(125,211,252,0.28); } blockquote { border-left: 3px solid #7dd3fc; padding: 10px 16px; color: #cbd5e1; font-style: italic; background: rgba(125,211,252,0.06); border-radius: 0 8px 8px 0; margin: 12px 0; font-size: 21px; } img { border-radius: 10px; box-shadow: 0 10px 28px rgba(0,0,0,0.55); border: 1px solid rgba(125,211,252,0.28); } section::after { color: #7dd3fc; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 15px; font-weight: 500; background: rgba(125,211,252,0.08); border: 1px solid rgba(125,211,252,0.2); padding: 4px 12px; border-radius: 6px; bottom: 22px; right: 28px; } .chip { display: inline-block; padding: 6px 14px; border-radius: 999px; background: rgba(125,211,252,0.08); border: 1px solid rgba(125,211,252,0.22); font-size: 18px; color: #e6eaf2; margin: 4px 6px; } .chip code { background: transparent; border: none; color: #7dd3fc; padding: 0 0 0 6px; font-size: 16px; } footer { color: #64748b; font-size: 14px; font-family: 'IBM Plex Sans', sans-serif; }"
footer: "Engine Failure & RUL Prediction · RCSSTEAP, Beihang University"
---

<!-- _class: lead -->
<!-- _paginate: false -->

<span class="eyebrow">Final Project · RCSSTEAP · Beihang University</span>

# Engine Failure &
# Remaining Useful Life
## Prediction with Physics-Informed Deep Learning

<br>

### NASA C-MAPSS Turbofan Dataset · CNN vs PINN

<br>

<span class="chip">Kodie Amo Kwame <code>LS2525226</code></span>
<span class="chip">Sumara Alfred Salifu <code>LS2525245</code></span>
<span class="chip">Peta Mimi Precious <code>LS2525255</code></span>

<!-- note: Jet engines fail. When they do, lives and millions of dollars are at stake. This final project predicts engine failure using physics-informed deep learning on NASA's turbofan dataset. We compare a standard Convolutional Neural Network against a Physics-Informed Neural Network, and show that embedding physical laws into the loss function produces safer, more accurate predictions of remaining useful life. -->

---

## The Dataset — NASA C-MAPSS

![bg right:50% fit](../output/sensor_trajectories.png)

**C-MAPSS** = *Commercial Modular Aero-Propulsion System Simulation*

- **4 fault datasets:** FD001 – FD004 (increasing complexity)
- **21 sensors:** temperatures, pressures, rotational speeds
- **3 operating settings:** altitude, Mach, throttle
- **100 training engines + 100 test engines** (FD001)
- Each engine runs until degradation-induced failure

<br>

> Sensors capture the gradual fingerprint of wear — hotter exhaust gas, lower core speed, rising fuel flow.

<!-- note: NASA's C-MAPSS dataset — Commercial Modular Aero-Propulsion System Simulation — spans four fault datasets, F D zero zero one through F D zero zero four, in rising complexity. Each engine is monitored with twenty one sensors and three operating settings. We focus on F D zero zero one: one hundred training engines and one hundred test engines, every one running until failure. -->

---

## What Is Remaining Useful Life?

![bg right:50% fit](../output/rul_trajectory_example.png)

**Remaining Useful Life (RUL)** — the number of operating cycles an engine has left before it fails.

- **Healthy phase:** RUL is effectively constant
- **Degradation phase:** RUL drops linearly
- We apply a **piecewise-linear cap at 125 cycles** so the model focuses on the *informative* regime
- RUL is the fundamental quantity for *predictive maintenance* and *condition-based overhaul*

<!-- note: Remaining Useful Life — abbreviated R U L — is the number of cycles an engine has left before failure. Early in life, sensors show almost no signal, so true R U L is effectively constant. Only the degradation phase is informative, so we cap R U L at one hundred twenty five cycles. This is the key quantity for predictive maintenance. -->

---

## Two Models, One Task

![bg right:45% fit](../output/network_architectures.png)

### 1. Baseline — 1D Convolutional Neural Network (CNN)
`Conv1D(16, k=5) → ReLU → GlobalAvgPool → Dense(32) → Dense(1)`
Pure data-driven; learns only from sensor windows.

<br>

### 2. Physics-Informed Neural Network (PINN)
`Dense(96) → ReLU → Dense(48) → ReLU → Dense(1)`
**+ physics loss** on adjacent-cycle pairs.

<!-- note: We built two models. A baseline one-dimensional Convolutional Neural Network — C N N — with sixteen filters, a Rectified Linear Unit, global average pooling, and two dense layers. And a Physics-Informed Neural Network — P I N N — a multi-layer perceptron that adds a physics loss on adjacent cycle pairs. -->

---

## The Physics Loss — Injecting Domain Knowledge

Applied on **adjacent-cycle window pairs** `(t, t+1)`:

### Monotonicity penalty
`L_mono = ReLU( RUL(t+1) − RUL(t) )²`
> RUL must never *increase* — a degraded engine cannot un-degrade.

### Drift penalty
`L_drift = ( RUL(t) − RUL(t+1) − 1 )²`
> RUL must drop by *exactly one cycle* per cycle of operation.

<br>

**Total loss:**  `L = L_MSE + 0.15 · L_mono + 0.05 · L_drift`

<!-- note: Two physics terms on adjacent cycle pairs. The monotonicity penalty discourages R U L going up — a degraded engine cannot un-degrade. The drift penalty enforces R U L dropping by exactly one per cycle, the literal definition. Total loss is mean squared error plus lambda zero point one five times monotonicity plus zero point zero five times drift. -->

---

## Training Setup

![bg right:50% fit](../output/training_curves.png)

- **Window length:** 30 cycles · **Stride:** 1
- **Training windows:** 14,207 · **Validation:** 3,524
- **Feature channels:** 18 (dropped 6 constant sensors)
- **CNN:** 18 epochs · **PINN:** 22 epochs
- **Total wall time:** **20.5 seconds** on CPU (pure NumPy)
- PINN converges to lower validation loss — the physics prior acts as a **free regularizer**

<!-- note: Sliding windows of thirty cycles produce fourteen thousand two hundred seven training windows and three thousand five hundred twenty four validation windows, across eighteen feature channels. Total training time: twenty point five seconds on CPU, using pure NumPy. The P I N N converges to lower validation loss — the physics prior acts as a free regularizer. -->

---

## Predictions vs Ground Truth

![bg left:48% fit](../output/pred_vs_actual_cnn.png)
![bg right:48% fit](../output/pred_vs_actual_pinn.png)

<br><br><br><br>

**Left — CNN:** predictions scatter; overshoots at low RUL
**Right — PINN:** hugs the diagonal, *especially* near end of life

<br>

> The end-of-life region is where *correct* predictions matter most — a missed warning there means an in-flight failure.

<!-- note: On one hundred unseen test engines, the C N N predictions scatter and overshoot at low R U L — optimistic exactly when pessimism matters. The P I N N predictions hug the diagonal tightly, especially near end of life. That is where a missed warning means in-flight failure, not just a scheduling inconvenience. -->

---

## Quantitative Results

![bg right:42% fit](../output/model_comparison.png)

|   | CNN | **PINN** | Improvement |
|---|---:|---:|---:|
| **RMSE** (Root MSE) | 18.68 | **14.64** | **−21.6 %** |
| **MAE** (Mean Abs Err) | 13.69 | **11.14** | **−18.6 %** |
| **NASA score** (asym.) | 1107.5 | **338.6** | **−69.4 %** |

<br>

> The **NASA score** penalizes *late* predictions ~4× harder than early ones — because in aviation, a late warning is far more costly. This is where physics helps most.

<!-- note: Root Mean Square Error drops from eighteen point six eight to fourteen point six four — a twenty one point six percent improvement. Mean Absolute Error drops from thirteen point six nine to eleven point one four. The N A S A score, which punishes late predictions four times harder than early ones, drops from eleven hundred seven to three hundred thirty eight — nearly seventy percent. This is where physics helps most. -->

---

## Error Analysis

![bg left:48% fit](../output/error_histogram.png)
![bg right:48% fit](../output/per_engine_rul.png)

<br><br><br><br>

- **Left:** error distribution — PINN is *tighter and more symmetric*, fewer large late-predictions
- **Right:** per-engine RUL trajectories — PINN tracks ground truth smoothly, without post-hoc filtering
- The monotonicity constraint eliminates almost all *rising* prediction segments

<!-- note: The P I N N error distribution is tighter and more symmetric, with fewer large late predictions — the errors N A S A's score punishes most. Per-engine R U L trajectories track ground truth smoothly, with no post-hoc filtering. The monotonicity constraint eliminates almost all rising prediction segments. -->

---

<!-- _class: lead -->
<!-- _paginate: false -->

<span class="eyebrow">Conclusion</span>

# Physics-Informed Learning
# = Safer, More Accurate RUL

<br>

### *A small network with the right prior beats a larger network with none.*

<br>

**Thank you**

<span class="chip">Kodie Amo Kwame</span>
<span class="chip">Sumara Alfred Salifu</span>
<span class="chip">Peta Mimi Precious</span>

<br>

*RCSSTEAP · Beihang University · Final Project*

<!-- note: The key takeaway: a small network with the right physical prior beats a larger network with none. Physics-informed learning cut Root Mean Square Error by over twenty percent, and the asymmetric N A S A score by nearly seventy percent — turning a compact neural network into a safer, more accurate estimator for condition-based engine maintenance. Thank you. -->
