"""Plotly/Matplotlib helpers for C-MAPSS exploratory analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


def non_constant_sensors(df: pd.DataFrame, eps: float = 1e-9) -> list[str]:
    cols = sensor_columns(df)
    out = []
    for c in cols:
        if float(df[c].std()) > eps:
            out.append(c)
    return out


def constant_sensors(df: pd.DataFrame, eps: float = 1e-9) -> list[str]:
    return [c for c in sensor_columns(df) if float(df[c].std()) <= eps]


def subsample_df(df: pd.DataFrame, max_rows: int = 60_000, seed: int = 0) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=seed)


def fig_correlation_heatmap(df: pd.DataFrame, title: str, max_rows: int = 50_000) -> go.Figure:
    cols = non_constant_sensors(df)
    if len(cols) < 2:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="Not enough varying sensors", showarrow=False)])
        return fig
    sub_full = df[cols].astype(float)
    sub = subsample_df(sub_full, max_rows=max_rows)
    corr = sub.corr()
    note = f" (n={len(sub):,} rows sampled)" if len(sub) < len(sub_full) else f" (n={len(sub):,})"
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=[c.replace("sensor_", "s") for c in cols],
            y=[c.replace("sensor_", "s") for c in cols],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title=title + note,
        height=560,
        margin=dict(l=60, r=20, t=50, b=60),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_max_cycle_per_unit(df: pd.DataFrame, title: str) -> go.Figure:
    g = df.groupby("unit")["cycle"].max().sort_values(ascending=True)
    fig = go.Figure(
        data=go.Bar(x=g.values, y=[str(u) for u in g.index], orientation="h", marker_color="#3b6ea5")
    )
    fig.update_layout(
        title=title,
        height=min(800, max(400, 12 * len(g))),
        xaxis_title="Last observed cycle",
        yaxis_title="Engine unit",
        margin=dict(l=80, r=20, t=50, b=40),
    )
    return fig


def fig_settings_2d(df: pd.DataFrame, title: str) -> go.Figure:
    sample = df if len(df) <= 8000 else df.sample(8000, random_state=0)
    fig = go.Figure(
        data=go.Scatter(
            x=sample["setting_1"],
            y=sample["setting_2"],
            mode="markers",
            marker=dict(size=4, color=sample["cycle"], colorscale="Viridis", opacity=0.5),
            text=sample["unit"],
            hovertemplate="setting1=%{x:.4f}<br>setting2=%{y:.4f}<br>cycle=%{marker.color:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Operational setting 1",
        yaxis_title="Operational setting 2",
        height=420,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def fig_sensor_std_bar(df: pd.DataFrame, title: str) -> go.Figure:
    cols = sensor_columns(df)
    stds = [float(df[c].std()) for c in cols]
    fig = go.Figure(
        data=go.Bar(x=[c.replace("sensor_", "s") for c in cols], y=stds, marker_color="#c44e52")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Sensor",
        yaxis_title="Std dev (full table)",
        height=400,
        margin=dict(l=50, r=20, t=50, b=80),
        xaxis=dict(tickangle=-45),
    )
    return fig


def fig_run_length_histogram(df: pd.DataFrame, title: str) -> go.Figure:
    mx = df.groupby("unit")["cycle"].max()
    fig = go.Figure(data=go.Histogram(x=mx.values, nbinsx=25, marker_color="#457b9d"))
    fig.update_layout(
        title=title,
        xaxis_title="Last observed cycle (engine life in dataset)",
        yaxis_title="Number of engines",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def fig_rul_histogram(rul: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(data=go.Histogram(x=rul, nbinsx=30, marker_color="#6a994e"))
    fig.update_layout(
        title=title,
        xaxis_title="True RUL (cycles)",
        yaxis_title="Count (engines)",
        height=400,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def fig_normalized_ensemble(
    df: pd.DataFrame,
    sensor_cols: list[str],
    title: str,
) -> go.Figure:
    """Mean sensor vs normalized engine life (cycle / max_cycle per unit)."""
    if not sensor_cols:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="No sensors", showarrow=False)])
        return fig
    rows = []
    for u, g in df.groupby("unit"):
        m = float(g["cycle"].max())
        if m < 1:
            continue
        gg = g.copy()
        gg["life"] = gg["cycle"] / m
        rows.append(gg)
    if not rows:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="No data", showarrow=False)])
        return fig
    long = pd.concat(rows, ignore_index=True)
    # Bin life for stable mean
    long["life_bin"] = (long["life"] * 99).round().astype(int).clip(0, 99)
    use_cols = sensor_cols[:3]
    nrows = len(use_cols)
    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=use_cols,
    )
    for i, s in enumerate(use_cols):
        agg = long.groupby("life_bin")[s].mean()
        x = (agg.index.astype(float) + 0.5) / 100.0
        r = i + 1
        fig.add_trace(
            go.Scatter(x=x, y=agg.values, mode="lines", name=s, showlegend=False),
            row=r,
            col=1,
        )
        fig.update_yaxes(title_text=s.replace("sensor_", "s"), row=r, col=1)
    fig.update_layout(height=200 * nrows + 60, title=title, margin=dict(l=50, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Normalized life (0=start, 1=last cycle)", row=nrows, col=1)
    return fig


def fig_pair_sensors_last_snapshot(df: pd.DataFrame, s1: str, s2: str, title: str) -> go.Figure:
    """Last-cycle snapshot per engine: useful for spread / fault mixing."""
    last = df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1)
    fig = go.Figure(
        data=go.Scatter(
            x=last[s1],
            y=last[s2],
            mode="markers",
            marker=dict(size=8, color=last["unit"], colorscale="Turbo", opacity=0.75),
            text=last["unit"],
            hovertemplate=f"{s1}=%{{x:.4f}}<br>{s2}=%{{y:.4f}}<br>unit=%{{text}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=s1,
        yaxis_title=s2,
        height=440,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def fig_sensor_cycle_correlation(df: pd.DataFrame, title: str, max_rows: int = 80_000) -> go.Figure:
    """Absolute Pearson correlation between each sensor and cycle index (fleet-wide exploratory trend)."""
    cols = non_constant_sensors(df)
    if not cols:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="No varying sensors", showarrow=False)])
        return fig
    sub = subsample_df(df, max_rows=max_rows)
    cy = sub["cycle"].astype(float)
    corrs: list[float] = []
    for c in cols:
        corrs.append(float(sub[c].astype(float).corr(cy)))
    abs_r = [abs(x) for x in corrs]
    note = f" (n={len(sub):,}" + (" sampled" if len(sub) < len(df) else "") + ")"
    fig = go.Figure(
        data=go.Bar(
            x=[c.replace("sensor_", "s") for c in cols],
            y=abs_r,
            marker_color="#2a9d8f",
            text=[f"{v:.3f}" for v in corrs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title + note,
        xaxis_title="Sensor",
        yaxis_title="|corr(sensor, cycle)|",
        height=460,
        margin=dict(l=50, r=20, t=50, b=90),
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def fig_settings_3d(df: pd.DataFrame, title: str, max_rows: int = 12_000) -> go.Figure:
    sub = subsample_df(df, max_rows=max_rows)
    fig = go.Figure(
        data=go.Scatter3d(
            x=sub["setting_1"],
            y=sub["setting_2"],
            z=sub["setting_3"],
            mode="markers",
            marker=dict(
                size=2,
                color=sub["cycle"],
                colorscale="Plasma",
                opacity=0.55,
                colorbar=dict(title="cycle"),
            ),
            text=sub["unit"],
            hovertemplate="s1=%{x:.4f}<br>s2=%{y:.4f}<br>s3=%{z:.4f}<br>cycle=%{marker.color:.0f}<extra></extra>",
        )
    )
    note = f" (sample n={len(sub):,})" if len(sub) < len(df) else f" (n={len(sub):,})"
    fig.update_layout(
        title=title + note,
        scene=dict(
            xaxis_title="Setting 1",
            yaxis_title="Setting 2",
            zaxis_title="Setting 3",
        ),
        height=520,
        margin=dict(l=0, r=0, t=50, b=20),
    )
    return fig


def fig_pca_last_snapshot(df: pd.DataFrame, title: str) -> go.Figure:
    """PCA (SVD) on standardized last-cycle sensor vector per engine."""
    cols = non_constant_sensors(df)
    if len(cols) < 3:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="Need >=3 varying sensors", showarrow=False)])
        return fig
    last = df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1)
    X = last[cols].values.astype(float)
    X = X - X.mean(axis=0)
    sdev = X.std(axis=0)
    sdev[sdev < 1e-12] = 1.0
    X = X / sdev
    _, sing, vt = np.linalg.svd(X, full_matrices=False)
    scores = X @ vt.T
    pc1 = scores[:, 0]
    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
    var = (sing**2) / max((sing**2).sum(), 1e-12)
    fig = go.Figure(
        data=go.Scatter(
            x=pc1,
            y=pc2,
            mode="markers",
            marker=dict(size=9, color=last["unit"], colorscale="Turbo", opacity=0.8),
            text=last["unit"],
            hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>unit=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title + f" — PC1={var[0]*100:.1f}% var, PC2={var[1]*100:.1f}% var",
        xaxis_title="PC1 (last snapshot)",
        yaxis_title="PC2 (last snapshot)",
        height=460,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def fig_train_test_last_overlay(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor: str,
    title: str,
) -> go.Figure:
    t_last = train_df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1)[sensor]
    te_last = test_df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1)[sensor]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=t_last, name="train (last cycle)", opacity=0.55, nbinsx=45))
    fig.add_trace(go.Histogram(x=te_last, name="test (last cycle)", opacity=0.55, nbinsx=45))
    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title=sensor,
        yaxis_title="Engine count",
        height=400,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def fig_rul_overview(rul: np.ndarray, title: str) -> go.Figure:
    x = np.arange(1, len(rul) + 1, dtype=float)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("RUL distribution (test engines)", "RUL vs file row index"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Histogram(x=rul, nbinsx=32, marker_color="#6a994e", showlegend=False), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rul,
            mode="markers",
            marker=dict(size=5, color=x, colorscale="Blues", opacity=0.65),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="True RUL (cycles)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Row / engine index in RUL file", row=1, col=2)
    fig.update_yaxes(title_text="RUL", row=1, col=2)
    fig.update_layout(title=title, height=420, margin=dict(l=50, r=20, t=60, b=40))
    return fig


def fig_normalized_ensemble_extended(
    df: pd.DataFrame,
    sensor_cols: list[str],
    title: str,
) -> go.Figure:
    """Mean sensor vs normalized engine life — up to four sensors (stacked)."""
    if not sensor_cols:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="No sensors", showarrow=False)])
        return fig
    use_cols = sensor_cols[:4]
    rows: list[pd.DataFrame] = []
    for u, g in df.groupby("unit"):
        m = float(g["cycle"].max())
        if m < 1:
            continue
        gg = g.copy()
        gg["life"] = gg["cycle"] / m
        rows.append(gg)
    if not rows:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="No data", showarrow=False)])
        return fig
    long = pd.concat(rows, ignore_index=True)
    long["life_bin"] = (long["life"] * 99).round().astype(int).clip(0, 99)
    nrows = len(use_cols)
    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=use_cols,
    )
    for i, s in enumerate(use_cols):
        agg = long.groupby("life_bin")[s].mean()
        xx = (agg.index.astype(float) + 0.5) / 100.0
        fig.add_trace(
            go.Scatter(x=xx, y=agg.values, mode="lines", name=s, showlegend=False),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=s.replace("sensor_", "s"), row=i + 1, col=1)
    fig.update_layout(
        height=min(900, 200 * nrows + 80),
        title=title,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Normalized life (0=start, 1=last cycle)", row=nrows, col=1)
    return fig
