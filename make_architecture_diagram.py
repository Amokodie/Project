"""Draw a side-by-side architecture diagram of the two models
(CNN, PINN) plus the shared training loop.
Saves output/network_architectures.png .
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "network_architectures.png")


def box(ax, x, y, w, h, text, color, edge="#0f172a", fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#0f172a", fontweight=weight)


def arrow(ax, x0, y0, x1, y1, color="#334155"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3))


def draw_stack(ax, title, layers, loss_rows, color_hint, cx=5.0, w=5.8, h=0.7, gap=0.18, top=9.4):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color="#0f172a", loc="left")
    xs = cx - w / 2
    y = top
    ys = []
    for text, color in layers:
        box(ax, xs, y - h, w, h, text, color, fontsize=8.5)
        ys.append((y - h / 2, y - h))
        y -= h + gap
    for i in range(len(layers) - 1):
        arrow(ax, cx, ys[i][1], cx, ys[i + 1][0] + h / 2 + 0.04)
    # loss rows
    y_top = ys[-1][1] - 0.3
    for i, (text, col, bold) in enumerate(loss_rows):
        box(ax, xs, y_top - (i + 1) * 0.75, w, 0.68, text, col, fontsize=8.5, bold=bold)
    if loss_rows:
        arrow(ax, cx, ys[-1][1], cx, y_top - 0.07)
    return xs, w, top, y_top


def draw_cnn(ax):
    layers = [
        ("Input window   (30 x 18)", "#e2e8f0"),
        ("Conv1D   F=16,  kernel K=5", "#fde68a"),
        ("ReLU", "#fca5a5"),
        ("Global Average Pool", "#bae6fd"),
        ("Dense (32)  +  ReLU", "#c4b5fd"),
        ("Dense (1)   -> RUL_hat", "#86efac"),
    ]
    loss_rows = [
        ("Loss = mean((RUL_hat - RUL_true)^2)   (MSE only)", "#fecaca", True),
    ]
    draw_stack(ax, "Model 1 - CNN  (2k weights)\nno physics", layers, loss_rows,
               color_hint="#f97316")


def draw_pinn(ax):
    layers = [
        ("Input window (30 x 18) -> flatten (540)", "#e2e8f0"),
        ("Dense (96)  +  ReLU", "#c4b5fd"),
        ("Dense (48)  +  ReLU", "#c4b5fd"),
        ("Dense (1)   -> RUL_hat", "#86efac"),
    ]
    loss_rows = [
        ("L_data  = mean((RUL_hat - RUL_true)^2)", "#fecaca", False),
        ("L_mono  = mean(ReLU(RUL(t+1)-RUL(t))^2)", "#fecaca", False),
        ("L_drift = mean((RUL(t)-RUL(t+1)-1)^2)", "#fecaca", False),
        ("Total = L_data + 0.15*L_mono + 0.05*L_drift", "#fca5a5", True),
    ]
    draw_stack(ax, "Model 2 - PINN  (57k weights)\nphysics loss, MLP body", layers, loss_rows,
               color_hint="#22d3ee")


def draw_training_loop(ax):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Training loop (identical for both models)", fontsize=12, color="#0f172a", loc="left")

    steps = [
        ("Init weights\n(He init)", "#e2e8f0"),
        ("Draw\nmini-batch\n(N=128)", "#fde68a"),
        ("Forward\npass", "#bae6fd"),
        ("Compute\nloss", "#fca5a5"),
        ("Back-\npropagation", "#c4b5fd"),
        ("Adam\nupdate", "#86efac"),
        ("Next batch /\nnext epoch", "#fde68a"),
    ]
    n = len(steps)
    box_w = 1.7
    box_h = 1.3
    gap = 0.25
    total_w = n * box_w + (n - 1) * gap
    x0 = (14 - total_w) / 2
    y0 = 1.5
    positions = []
    for i, (text, color) in enumerate(steps):
        xi = x0 + i * (box_w + gap)
        box(ax, xi, y0, box_w, box_h, text, color, fontsize=8)
        positions.append(xi + box_w / 2)

    for i in range(n - 1):
        arrow(ax, x0 + (i + 1) * (box_w + gap) - gap,
              y0 + box_h / 2,
              x0 + (i + 1) * (box_w + gap),
              y0 + box_h / 2)

    ax.annotate("", xy=(positions[1], y0 + box_h + 0.35),
                xytext=(positions[-1], y0 + box_h + 0.35),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text((positions[1] + positions[-1]) / 2, y0 + box_h + 0.55,
            "repeat until the dataset has been seen 1 time = one EPOCH",
            ha="center", fontsize=9, color="#334155")


def draw_result_table(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Final test-set numbers (FD001)", fontsize=11, color="#0f172a", loc="left")
    cells = [
        ["Model",           "RMSE",   "MAE",   "NASA score"],
        ["CNN (baseline)",  "18.68",  "13.69", "1107.5"],
        ["PINN",            "14.43",  "10.99", " 323.8"],
    ]
    colors = ["#e2e8f0", "#ffedd5", "#cffafe"]
    n_rows = len(cells)
    n_cols = len(cells[0])
    w = 2.3
    h = 0.7
    x0 = 0.6
    y0 = 3.4
    for r in range(n_rows):
        for c in range(n_cols):
            color = "#f1f5f9" if r == 0 else colors[r]
            bold = r == 0 or r == 2
            box(ax, x0 + c * w, y0 - r * h, w, h, cells[r][c], color, fontsize=10, bold=bold)
    ax.text(0.6, 0.8, "Lower is better on every metric.", fontsize=10, color="#334155")
    ax.text(0.6, 0.3, "PINN wins overall on RMSE, MAE and NASA score.",
            fontsize=10, color="#334155", style="italic")


def main() -> None:
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 2.4],
                          width_ratios=[1, 1, 1.1], hspace=0.25, wspace=0.15)
    ax_cnn = fig.add_subplot(gs[0, 0])
    ax_pinn = fig.add_subplot(gs[0, 1])
    ax_tab = fig.add_subplot(gs[0, 2])
    ax_train = fig.add_subplot(gs[1, :])

    draw_cnn(ax_cnn)
    draw_pinn(ax_pinn)
    draw_result_table(ax_tab)
    draw_training_loop(ax_train)

    fig.suptitle(
        "How the two models are built\n"
        "C-MAPSS FD001  -  RUL prediction  -  pure NumPy",
        fontsize=14, color="#0f172a", y=0.99,
    )
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
