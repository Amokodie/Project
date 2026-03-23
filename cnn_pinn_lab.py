"""Illustrative CNN vs PINN comparison charts for the PHM dashboard (not trained models)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fig_training_loss_curves(epochs: int = 120) -> go.Figure:
    """Synthetic validation loss vs epoch — CNN vs PINN (discussion only)."""
    e = np.arange(0, epochs + 1)
    rng = np.random.default_rng(42)
    # Illustrative: PINN curve often drops faster when physics residual is informative
    cnn = 0.62 * np.exp(-e / 38.0) + 0.026 + rng.normal(0, 0.004, len(e))
    pinn = 0.58 * np.exp(-e / 28.0) + 0.014 + rng.normal(0, 0.004, len(e))
    cnn = np.maximum(cnn, 0.017)
    pinn = np.maximum(pinn, 0.011)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e, y=cnn, mode="lines", name="1D-CNN window baseline", line=dict(color="#f97316", width=2.5)))
    fig.add_trace(go.Scatter(x=e, y=pinn, mode="lines", name="PINN + wear residual", line=dict(color="#22d3ee", width=2.5)))
    fig.update_layout(
        title="Illustrative validation loss (RMSE-style) — not from your trained weights",
        xaxis_title="Epoch",
        yaxis_title="Loss (arbitrary units)",
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def fig_radar_cnn_vs_pinn() -> go.Figure:
    """Qualitative radar — same categories, different profiles."""
    cats = [
        "Multi-regime<br>(FD004)",
        "Data<br>efficiency",
        "Interpretability",
        "Implementation<br>effort",
        "Edge<br>latency",
        "Physics<br>consistency",
    ]
    cnn = [88, 55, 42, 72, 78, 38]
    pinn = [62, 82, 85, 58, 70, 90]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=cnn + [cnn[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="CNN / Conv backbone",
            line_color="#f97316",
            fillcolor="rgba(249,115,22,0.25)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=pinn + [pinn[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name="PINN-style",
            line_color="#22d3ee",
            fillcolor="rgba(34,211,238,0.22)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.3)")),
        showlegend=True,
        height=480,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title="Qualitative design radar — scores are for discussion, not measured",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def fig_loss_components_bar() -> go.Figure:
    """Stacked bar: typical loss terms (illustrative proportions)."""
    df = pd.DataFrame(
        {
            "Model": ["1D-CNN RUL head", "PINN (hybrid)"],
            "Data (supervised)": [78, 45],
            "Physics / residual": [0, 38],
            "Regularization": [22, 17],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Supervised data", x=df["Model"], y=df["Data (supervised)"], marker_color="#38bdf8")
    )
    fig.add_trace(go.Bar(name="Physics residual", x=df["Model"], y=df["Physics / residual"], marker_color="#a78bfa"))
    fig.add_trace(go.Bar(name="Regularization / priors", x=df["Model"], y=df["Regularization"], marker_color="#64748b"))
    fig.update_layout(
        barmode="stack",
        title="Illustrative loss budget split (%)",
        yaxis_title="Relative emphasis (%)",
        height=380,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def fig_sensor_window_vs_physics() -> go.Figure:
    """Dual-axis style subplot: receptive field vs physics constraints."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CNN: local time receptive field", "PINN: wear + residual"))
    t = np.linspace(0, 1, 80)
    # fake sensor trace
    sig = 0.3 * np.sin(6 * np.pi * t) + 0.15 * t**2 + np.random.default_rng(1).normal(0, 0.02, len(t))
    fig.add_trace(
        go.Scatter(x=t, y=sig, line=dict(color="#f97316"), fill="tozeroy", fillcolor="rgba(249,115,22,0.15)", name="Window"),
        row=1,
        col=1,
    )
    wear = 0.02 * np.exp(2.2 * t)
    fig.add_trace(go.Scatter(x=t, y=wear, line=dict(color="#22d3ee", width=3), name="Wear prior"), row=1, col=2)
    fig.update_xaxes(title_text="Normalized time in window", row=1, col=1)
    fig.update_xaxes(title_text="Cycle / life proxy", row=1, col=2)
    fig.update_yaxes(title_text="Signal", row=1, col=1)
    fig.update_yaxes(title_text="Residual magnitude", row=1, col=2)
    fig.update_layout(
        height=360,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig
