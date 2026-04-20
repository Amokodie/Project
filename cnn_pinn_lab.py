"""CNN vs Transformer vs PINN — illustrative charts (not trained on your data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Neutral defaults; Streamlit applies `plotly_theme.apply_plotly_theme` in app.py.
_PAPER = "rgba(0,0,0,0)"
_PLOT = "rgba(15,23,42,0.4)"


def fig_training_loss_curves(epochs: int = 120) -> go.Figure:
    """Synthetic validation loss — CNN, Transformer encoder, PINN."""
    e = np.arange(0, epochs + 1)
    rng = np.random.default_rng(42)
    cnn = 0.62 * np.exp(-e / 38.0) + 0.026 + rng.normal(0, 0.004, len(e))
    tfm = 0.59 * np.exp(-e / 32.0) + 0.019 + rng.normal(0, 0.004, len(e))
    pinn = 0.58 * np.exp(-e / 28.0) + 0.014 + rng.normal(0, 0.004, len(e))
    cnn = np.maximum(cnn, 0.017)
    tfm = np.maximum(tfm, 0.014)
    pinn = np.maximum(pinn, 0.011)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=e, y=cnn, mode="lines", name="1D-CNN windows", line=dict(color="#f97316", width=2.2))
    )
    fig.add_trace(
        go.Scatter(
            x=e,
            y=tfm,
            mode="lines",
            name="Temporal Transformer",
            line=dict(color="#a78bfa", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(x=e, y=pinn, mode="lines", name="PINN + wear residual", line=dict(color="#22d3ee", width=2.2))
    )
    fig.update_layout(
        title="Illustrative validation loss — CNN vs Transformer vs PINN (not your trained run)",
        xaxis_title="Epoch",
        yaxis_title="Loss (arbitrary units)",
        height=440,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def fig_radar_three_way() -> go.Figure:
    """Qualitative radar — CNN vs Transformer vs PINN."""
    cats = [
        "Multi-regime<br>(FD004)",
        "Long-range<br>dependencies",
        "Data<br>efficiency",
        "Interpretability",
        "Edge<br>deploy",
        "Physics<br>consistency",
    ]
    cnn = [86, 58, 52, 44, 80, 36]
    tfm = [84, 88, 48, 46, 52, 40]
    pinn = [60, 55, 84, 88, 68, 92]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=cnn + [cnn[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="1D-CNN",
            line_color="#f97316",
            fillcolor="rgba(249,115,22,0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=tfm + [tfm[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="Transformer",
            line_color="#a78bfa",
            fillcolor="rgba(167,139,250,0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=pinn + [pinn[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="PINN-style",
            line_color="#22d3ee",
            fillcolor="rgba(34,211,238,0.18)",
        )
    )
    grid = "rgba(148,163,184,0.35)"
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor=grid)),
        showlegend=True,
        height=500,
        paper_bgcolor=_PAPER,
        title="Qualitative comparison — discussion scores only",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def fig_loss_components_bar() -> go.Figure:
    """Three illustrative model families."""
    df = pd.DataFrame(
        {
            "Model": ["1D-CNN", "Temporal Transformer", "PINN"],
            "Supervised": [76, 70, 42],
            "Attention / seq": [0, 22, 0],
            "Physics residual": [0, 0, 40],
            "Regularization": [24, 8, 18],
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Supervised RUL / fit", x=df["Model"], y=df["Supervised"], marker_color="#38bdf8"))
    fig.add_trace(go.Bar(name="Attention / sequence prior", x=df["Model"], y=df["Attention / seq"], marker_color="#a78bfa"))
    fig.add_trace(go.Bar(name="Physics residual", x=df["Model"], y=df["Physics residual"], marker_color="#34d399"))
    fig.add_trace(go.Bar(name="Regularization", x=df["Model"], y=df["Regularization"], marker_color="#64748b"))
    fig.update_layout(
        barmode="stack",
        title="Illustrative loss-term emphasis (%)",
        yaxis_title="Relative emphasis",
        height=400,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def fig_sensor_window_physics_attention() -> go.Figure:
    """CNN window vs PINN vs fake attention weights (Transformer intuition)."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("CNN: local window", "Transformer: attention mass (illustrative)", "PINN: wear channel"),
    )
    t = np.linspace(0, 1, 64)
    rng = np.random.default_rng(1)
    sig = 0.3 * np.sin(6 * np.pi * t) + 0.15 * t**2 + rng.normal(0, 0.02, len(t))
    fig.add_trace(
        go.Scatter(x=t, y=sig, line=dict(color="#f97316"), fill="tozeroy", fillcolor="rgba(249,115,22,0.12)", showlegend=False),
        row=1,
        col=1,
    )
    pos = np.arange(32)
    attn = np.exp(-0.5 * ((pos - 22) / 5.0) ** 2)
    attn = attn / attn.sum()
    fig.add_trace(
        go.Bar(x=pos, y=attn, marker_color="#a78bfa", showlegend=False),
        row=1,
        col=2,
    )
    wear = 0.02 * np.exp(2.2 * t)
    fig.add_trace(go.Scatter(x=t, y=wear, line=dict(color="#22d3ee", width=2), showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text="Time in window", row=1, col=1)
    fig.update_xaxes(title_text="Position (token idx)", row=1, col=2)
    fig.update_xaxes(title_text="Life proxy", row=1, col=3)
    fig.update_yaxes(title_text="Signal", row=1, col=1)
    fig.update_yaxes(title_text="Weight", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=3)
    fig.update_layout(
        height=380,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
