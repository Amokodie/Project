"""
Assignment 2 — NASA C-MAPSS Turbofan PHM Streamlit Dashboard.

Run: streamlit run app.py
Set folder to your CMAPSSData directory (train/test/RUL files) or env CMAPSS_DATA_DIR.
"""

from __future__ import annotations

import base64
import io
import os

from PIL import Image

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from analysis_pdf import build_cmapss_brief_pdf
from cnn_pinn_lab import (
    fig_loss_components_bar,
    fig_radar_three_way,
    fig_sensor_window_physics_attention,
    fig_training_loss_curves,
)
from cmapss_data import (
    FD_SCENARIO_INFO,
    cmapss_file,
    default_local_test_path,
    list_available_datasets,
    load_cmapss_table,
    load_rul,
    load_test_fd001,
    readme_path,
    resolve_cmapss_root,
    rul_file,
    sensor_labels_for_ui,
)
from eda_charts import (
    constant_sensors,
    fig_correlation_heatmap,
    fig_max_cycle_per_unit,
    fig_normalized_ensemble_extended,
    fig_pair_sensors_last_snapshot,
    fig_pca_last_snapshot,
    fig_rul_overview,
    fig_run_length_histogram,
    fig_sensor_cycle_correlation,
    fig_sensor_std_bar,
    fig_settings_2d,
    fig_settings_3d,
    fig_train_test_last_overlay,
    non_constant_sensors,
    sensor_columns,
)
from ui_theme import hero_engineering_ribbon, inject_engineering_theme, plotly_template

st.set_page_config(
    page_title="C-MAPSS PHM Dashboard",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ASSETS_README = os.path.join(_ASSET_DIR, "readme_cmapss.txt")
# Optional: place the PHM08 paper PDF here for an extra download button (same citation as readme).
OPTIONAL_PAPER_PDF = os.path.join(_ASSET_DIR, "PHM08_Damage_Propagation_Modeling.pdf")
LOGO_BEIHANG = os.path.join(_ASSET_DIR, "beihang_university_logo.png")
LOGO_RCSSTEAP = os.path.join(_ASSET_DIR, "rcssteap_logo.png")


def _logo_on_white_tile(path: str, max_height_px: int = 140) -> None:
    """Show a PNG on a white rounded tile (same treatment for both institution logos)."""
    if not os.path.isfile(path):
        return
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            composed = Image.alpha_composite(bg, im).convert("RGB")
            buf = io.BytesIO()
            composed.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    # components.html reliably renders data-URI images on dark themes
    components.html(
        f"""
<div style="background:#ffffff;border-radius:10px;padding:10px 14px;text-align:center;box-sizing:border-box;">
  <img src="data:image/png;base64,{b64}"
       style="max-height:{max_height_px}px;width:100%;object-fit:contain;display:block;margin:0 auto;" alt="" />
</div>
        """,
        height=max_height_px + 36,
    )


@st.cache_data(show_spinner=True)
def _load_cmapss_bundle(root: str, fd: str):
    """Load train, test, and RUL if files exist under root."""
    fd = fd.upper()
    train_path = cmapss_file(root, "train", fd)
    test_path = cmapss_file(root, "test", fd)
    rul_p = rul_file(root, fd)
    train_df = load_cmapss_table(train_path) if os.path.isfile(train_path) else None
    test_df = load_cmapss_table(test_path) if os.path.isfile(test_path) else None
    rul_arr = None
    if os.path.isfile(rul_p):
        try:
            rul_arr = load_rul(rul_p)
        except Exception:
            rul_arr = None
    return train_df, test_df, rul_arr, train_path, test_path, rul_p


def render_authors_banner():
    """Institution logos + team (compact header)."""
    with st.container(border=True):
        c_logo1, c_logo2, c_text = st.columns([1.15, 1.15, 2.5])
        with c_logo1:
            if os.path.isfile(LOGO_BEIHANG):
                _logo_on_white_tile(LOGO_BEIHANG, max_height_px=150)
            else:
                st.caption("Add `assets/beihang_university_logo.png`")
        with c_logo2:
            if os.path.isfile(LOGO_RCSSTEAP):
                _logo_on_white_tile(LOGO_RCSSTEAP, max_height_px=150)
            else:
                st.caption("Add `assets/rcssteap_logo.png`")
        with c_text:
            st.markdown(
                """
**Beihang University** · *Beijing University of Aeronautics and Astronautics*  
**Regional Centre for Space Science and Technology Education in Asia and the Pacific (RCSSTEAP), China**

**Team:** Kodie Amo Kwame (LS2525226) · Sumara Alfred Salifu (LS2525245) · Peta Mimi Precious (LS2525255)
                """
            )


def pillar_intro():
    st.title("NASA C-MAPSS Turbofan Engine — PHM Dashboard")
    hero_engineering_ribbon(st.session_state.get("ui_theme", "dark"))
    render_authors_banner()
    st.markdown(
        """
This dashboard is structured around **three engineering pillars** for briefing a manager on
**Remaining Useful Life (RUL)** and fleet health monitoring:

1. **Physical system insight** — What the turbofan is, what we measure, and how degradation shows up in time series.  
2. **Model trade-offs** — When a data-driven sequence model (Transformer/CNN) fits, versus when **physics-informed** structure (PINN) is justified.  
3. **Compute & data reality** — Data quality (MNAR, drift, rare faults) and deployment cost (GEMM, KV cache, quantization).

Use the sidebar to point at your **CMAPSSData** folder, pick **FD001–FD004**, then explore **fleet trajectories**, **EDA plots**, and download the **PDF brief** and **readme** text.
        """
    )


def section_downloads(
    root: str,
    fd: str,
    train_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    rul: np.ndarray | None,
):
    st.subheader("Documentation & downloads")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        readme_disk = readme_path(root)
        if readme_disk and os.path.isfile(readme_disk):
            with open(readme_disk, "rb") as f:
                st.download_button(
                    label="Download NASA readme.txt (from your CMAPSS folder)",
                    data=f.read(),
                    file_name="readme.txt",
                    mime="text/plain",
                    key="dl_readme_cmapss",
                )
        elif os.path.isfile(ASSETS_README):
            with open(ASSETS_README, "rb") as f:
                st.download_button(
                    label="Download readme copy (bundled)",
                    data=f.read(),
                    file_name="readme_cmapss.txt",
                    mime="text/plain",
                    key="dl_readme_bundled",
                )
    with c2:
        pdf_bytes = build_cmapss_brief_pdf(fd, root, train_df, test_df, rul)
        st.download_button(
            label="Download analysis brief (PDF)",
            data=pdf_bytes,
            file_name=f"CMAPSS_{fd}_PHM_brief.pdf",
            mime="application/pdf",
            key="dl_pdf_brief",
        )
    with c3:
        if os.path.isfile(OPTIONAL_PAPER_PDF):
            with open(OPTIONAL_PAPER_PDF, "rb") as f:
                st.download_button(
                    label="Download PHM08 paper (PDF)",
                    data=f.read(),
                    file_name=os.path.basename(OPTIONAL_PAPER_PDF),
                    mime="application/pdf",
                    key="dl_phm08_paper",
                )
        else:
            st.caption(
                "Optional: add the conference paper PDF as "
                "`assets/PHM08_Damage_Propagation_Modeling.pdf` to enable a second download."
            )
    with c4:
        st.caption(
            "The brief PDF is generated in-app from your loaded tables and cites Saxena et al. PHM08. "
            "The NASA text bundle does not ship a PDF."
        )


def tab_fleet_overview(df: pd.DataFrame, data_caption: str):
    st.subheader("Fleet overview — degradation trajectories")
    st.markdown(
        """
**System context (C-MAPSS):** We monitor a **fleet of high-thrust turbofan engines** simulated in C-MAPSS.
The goal is **RUL prediction** by tracking degradation in subsystems such as the **high-pressure compressor (HPC)** and hot section,
using **26 channels per cycle**: three **operational settings** and **21 sensor measurements** (temperatures, pressures, speeds, flows).

Below, pick engines and sensors to visualize trajectories. Sensor names follow common C-MAPSS naming in the literature.
        """
    )
    labels = sensor_labels_for_ui()
    sensor_cols = sensor_columns(df)
    default_y1 = "sensor_4" if "sensor_4" in sensor_cols else sensor_cols[0]
    default_y2 = "sensor_11" if "sensor_11" in sensor_cols else sensor_cols[min(10, len(sensor_cols) - 1)]

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        units = sorted(df["unit"].unique().tolist())
        pick = st.multiselect("Engines (unit id)", units, default=units[: min(5, len(units))])
    with c2:
        y1 = st.selectbox("Primary series (e.g. LPT outlet proxy)", sensor_cols, index=sensor_cols.index(default_y1))
        y2 = st.selectbox("Secondary series (e.g. Ps30 proxy)", sensor_cols, index=sensor_cols.index(default_y2))
    with c3:
        st.caption(labels.get(y1, y1))
        st.caption(labels.get(y2, y2))

    sub = df[df["unit"].isin(pick)].copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(labels.get(y1, y1), labels.get(y2, y2)),
    )
    for u in pick:
        g = sub[sub["unit"] == u]
        fig.add_trace(go.Scatter(x=g["cycle"], y=g[y1], mode="lines", name=f"U{int(u)}"), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=g["cycle"], y=g[y2], mode="lines", name=f"U{int(u)}", showlegend=False),
            row=2,
            col=1,
        )
    fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40), legend_title_text="Engine")
    fig.update_xaxes(title_text="Flight cycle", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Reading the plot:** Rising temperature-like traces and falling pressure-margin proxies often accompany "
        "late-life behavior in these simulations—tie chosen sensors to **physics** (HPC, combustor, turbine)."
    )
    st.caption(data_caption)


def tab_eda(
    fd: str,
    train_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    rul: np.ndarray | None,
):
    st.subheader("Exploratory analysis — plots from your C-MAPSS files")
    info = FD_SCENARIO_INFO.get(fd.upper(), {})
    st.markdown(
        f"""
These views use **real columns** from **{fd}** where available. Training runs to failure; test is **censored** and matches **RUL_{fd[-3:]}.txt**.
Sensors with **near-zero variance** are listed below and **excluded** from correlation / PCA where noted.
        """
    )
    if info:
        st.info(
            f"**NASA scenario ({fd}):** operating conditions = **{info.get('conditions', '?')}**; "
            f"faults = **{info.get('faults', '?')}**. "
            f"Readme counts: ~**{info.get('train_engines', '?')}** train engines, ~**{info.get('test_engines', '?')}** test engines."
        )

    eda_train = st.checkbox("Include training split in EDA tables/plots", value=train_df is not None)
    base = train_df if eda_train and train_df is not None else test_df
    if base is None:
        base = train_df or test_df
    if base is None:
        st.warning("No train/test dataframe loaded.")
        return

    split_label = "train" if eda_train and train_df is not None else "test"
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rows (EDA base)", f"{len(base):,}")
    m2.metric("Engines", f"{base['unit'].nunique()}")
    m3.metric("Varying sensors", len(non_constant_sensors(base)))
    m4.metric("Constant sensors", len(constant_sensors(base)))
    m5.metric("Train rows", f"{len(train_df):,}" if train_df is not None else "—")
    m6.metric("RUL labels", len(rul) if rul is not None else 0)

    const_list = constant_sensors(base)
    with st.expander("Constant (flat) sensors — often dropped in papers", expanded=False):
        if const_list:
            st.write(", ".join(const_list))
        else:
            st.write("None detected at default tolerance.")

    st.markdown("#### Correlation matrix (Pearson) — varying sensors only")
    st.caption("Large tables are **row-sampled** for speed (see plot title).")
    st.plotly_chart(fig_correlation_heatmap(base, f"{fd} — sensor–sensor | {split_label}"), use_container_width=True)

    st.markdown("#### Trend strength: |corr(sensor, cycle)| (fleet-wide)")
    st.caption("Higher values suggest the sensor tracks **cycle time** in this split (useful RUL feature candidates).")
    st.plotly_chart(fig_sensor_cycle_correlation(base, f"{fd} | {split_label}"), use_container_width=True)

    st.markdown("#### Run length — last observed cycle per engine")
    if train_df is not None and test_df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_max_cycle_per_unit(train_df, f"{fd} — training (run-to-failure)"), use_container_width=True)
        with c2:
            st.plotly_chart(fig_max_cycle_per_unit(test_df, f"{fd} — test (censored)"), use_container_width=True)
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(
                fig_run_length_histogram(train_df, f"{fd} — run length distribution (train)"),
                use_container_width=True,
            )
        with h2:
            st.plotly_chart(
                fig_run_length_histogram(test_df, f"{fd} — run length distribution (test)"),
                use_container_width=True,
            )
    else:
        st.plotly_chart(fig_max_cycle_per_unit(base, f"{fd} — last cycle per engine"), use_container_width=True)
        st.plotly_chart(fig_run_length_histogram(base, f"{fd} — run length distribution"), use_container_width=True)

    st.markdown("#### Operational settings")
    cset1, cset2 = st.columns(2)
    with cset1:
        st.plotly_chart(fig_settings_2d(base, f"{fd} — settings 1 vs 2 (color=cycle)"), use_container_width=True)
    with cset2:
        st.plotly_chart(fig_settings_3d(base, f"{fd} — settings 1–3 (3D)"), use_container_width=True)

    st.markdown("#### Sensor variability (global std per channel)")
    st.plotly_chart(fig_sensor_std_bar(base, f"{fd} — per-sensor std"), use_container_width=True)

    if train_df is not None and test_df is not None:
        st.markdown("#### Train vs test — last-cycle value (same sensor)")
        v_opts = non_constant_sensors(train_df) or sensor_columns(train_df)
        cmp_s = st.selectbox("Sensor for train/test comparison", v_opts, index=min(3, len(v_opts) - 1), key="tt_cmp")
        st.plotly_chart(
            fig_train_test_last_overlay(train_df, test_df, cmp_s, f"{fd} — {cmp_s} at last observed cycle"),
            use_container_width=True,
        )

    if rul is not None and len(rul):
        st.markdown("#### True RUL (test engines)")
        st.plotly_chart(fig_rul_overview(rul, f"{fd} — RUL_FD{fd[-3:]}.txt"), use_container_width=True)

    st.markdown("#### Ensemble degradation shape — mean vs normalized engine life")
    v_all = non_constant_sensors(base)
    default_pick = v_all[:4] if len(v_all) >= 4 else v_all
    if not default_pick:
        default_pick = sensor_columns(base)[:4]
    pick_life = st.multiselect(
        "Sensors to average (up to 4)",
        options=v_all or sensor_columns(base),
        default=default_pick[:4],
        max_selections=4,
        key="ens_pick",
    )
    if pick_life:
        st.plotly_chart(
            fig_normalized_ensemble_extended(base, pick_life, f"{fd} — fleet-mean trajectory | {split_label}"),
            use_container_width=True,
        )

    st.markdown("#### PCA — last snapshot (engines as points)")
    st.caption("Standardized **varying** sensors at each engine’s **last** cycle; PC axes from SVD.")
    st.plotly_chart(fig_pca_last_snapshot(base, f"{fd} | {split_label}"), use_container_width=True)

    s_opts = non_constant_sensors(base)
    if len(s_opts) >= 2:
        st.markdown("#### Last-cycle snapshot — pick two sensors")
        s1 = st.selectbox("X sensor", s_opts, index=0, key="pair_x")
        s2 = st.selectbox("Y sensor", s_opts, index=min(1, len(s_opts) - 1), key="pair_y")
        st.plotly_chart(
            fig_pair_sensors_last_snapshot(base, s1, s2, f"{fd} — final point per engine"),
            use_container_width=True,
        )


def tab_data_audit():
    st.subheader('Data audit — Session 4 style "engineering dimensions"')
    st.markdown(
        """
**Quality dominates quantity.** Use this view as a **structured audit** of the NASA runs before trusting any ML model.

- **Completeness (MNAR risk):** Sensors can fail **during** fault windows—missingness may be **Missing Not At Random (MNAR)** and can blind the model at the worst time.  
- **Consistency:** **Calibration drift** over hundreds of operating hours can look like health change—needs reconciliation to a **physics / baseline** model.  
- **Rare faults:** A huge share of cycles are **nominal**; without **decoupled training** or **safety-weighted loss**, standard objectives may **under-weight** critical failure modes.

Scores below are **illustrative** for discussion—replace with your measured metrics from exploratory analysis.
        """
    )
    dimensions = ["Completeness", "Consistency", "Accuracy", "Timeliness", "Relevance"]
    scores = [62, 58, 72, 80, 85]
    notes = [
        "MNAR risk when sensors drop during transients",
        "Drift vs. real degradation (recalibration)",
        "Label noise / sensor noise trade-offs",
        "Latency vs. refresh rate for monitoring",
        "Features aligned to failure modes (HPC, fan)",
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores + [scores[0]],
            theta=dimensions + [dimensions[0]],
            fill="toself",
            name="Audit score",
            hovertemplate="%{theta}: %{r}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=480,
        margin=dict(l=40, r=40, t=40, b=40),
        title="Radar — five engineering dimensions (demo scores)",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(
            pd.DataFrame({"Dimension": dimensions, "Score (0–100)": scores, "Discussion hook": notes}),
            hide_index=True,
            use_container_width=True,
        )


def tab_architecture():
    st.subheader("Architecture selection — Transformers vs PINNs vs hybrid")
    st.markdown(
        """
**Traditional Transformers / CNNs:** Strong when you can exploit **large, diverse regimes** (e.g. FD004 with multiple flight conditions) to learn
rich temporal patterns **without** hand-deriving degradation PDEs.

**Physics-Informed Neural Networks (PINNs):** Attractive in **data-scarce** settings: embed structure such as **wear laws**
(e.g. \\(w = A e^{B t}\\) forms) directly in the loss alongside data fidelity—papers report regimes where PINNs can reach comparable error with **orders-of-magnitude less labeled data**
(the exact factor depends on the study; treat **250×** as a **literature talking point**, not a guaranteed constant).

**2024 frontier (discussion):** For **engine digital twins**, teams increasingly discuss **hybrids**—graph-structured dependencies (**GNN**),
sequence modeling (**Transformer**), and **physics constraints** (**PINN**) in one stack—your dashboard should position this as a **risk-aware** design choice, not hype.
        """
    )

    matrix = pd.DataFrame(
        {
            "Approach": ["Pure Transformer/CNN", "PINN (wear / residual laws)", "Hybrid GNN + Transformer + PINN"],
            "Data need": ["High (benefits from coverage)", "Lower (structure substitutes data)", "Medium–high (modular)"],
            "Interpretability": ["Low–medium", "High (embedded structure)", "High (selective physics)"],
            "Regulatory story": ["Black-box risk", "Stronger assurance path", "Balanced / staged rollout"],
            "Fit for C-MAPSS": ["FD00x with regime diversity", "Sparse labels / few engines", "Fleet twin + subsystem graphs"],
        }
    )
    st.markdown("#### Decision matrix (qualitative)")
    st.dataframe(matrix, hide_index=True, use_container_width=True)

    st.markdown("#### Why PINN can be “safer” than a pure black-box for aerospace")
    # Mermaid 10: avoid raw () / : / + in unquoted labels; use quoted strings.
    mermaid = """
flowchart TD
    A["Fleet time series - 26 ch per cycle"] --> B{"Data budget?"}
    B -->|large diverse data| C["Seq. model Transformer or CNN"]
    B -->|sparse engines| D["PINN data plus physics loss"]
    C --> E["Residual vs physics baselines"]
    D --> F["Wear in loss"]
    E --> G["Hybrid GNN Transformer PINN"]
    F --> G
    G --> H["Monitoring assurance drift faults"]
    """
    components.html(
        f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head><body>
<div class="mermaid">{mermaid}</div>
<script>mermaid.initialize({{startOnLoad:true, theme:'dark'}});</script>
</body></html>
        """,
        height=420,
    )


def tab_cnn_pinn_lab():
    tpl = plotly_template(st.session_state.get("ui_theme", "dark"))
    st.subheader("Deep learning lab — CNN · Transformer · PINN")
    st.markdown(
        """
**Turbofan RUL** can be approached with **1D convolutions** (local cycles), **Transformers** (global attention over a sequence of sensor snapshots),
or **PINNs** (supervised fit + physics / wear residuals). Curves and radars are **illustrative**—swap in your own training logs when available.
        """
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("1D-CNN", "Local windows", "fast inference")
    k2.metric("Transformer", "Self-attention", "long context")
    k3.metric("PINN", "Physics loss", "fewer labels?")
    k4.metric("FD004", "Multi-regime", "data-hungry")
    k5.metric("KV cache", "Transformer", "see Compute tab")

    st.markdown("#### Loss sketches (compact)")
    st.latex(
        r"""
        \mathcal{L}_{\text{CNN}} = \frac{1}{N}\sum_i \bigl( \hat{y}_i - y_i \bigr)^2 + \lambda \lVert W \rVert_2^2
        """
    )
    st.latex(
        r"""
        \mathcal{L}_{\text{Trans}} \approx \sum_i \bigl( \hat{y}_i - y_i \bigr)^2
        + \beta \sum_{l,h,i,j} \mathrm{Attn}_{h,l}(i,j)^2 \quad \text{(optional attention regularizer)}
        """
    )
    st.latex(
        r"""
        \mathcal{L}_{\text{PINN}} \approx \underbrace{\sum_j \bigl( s_j - \hat{s}_j \bigr)^2}_{\text{sensor fit}}
        + \lambda_1 \underbrace{\bigl( \partial_t \hat{w} - f_{\text{wear}}(t;\theta) \bigr)^2}_{\text{wear / residual}}
        + \lambda_2 \,\lVert \phi \rVert^2
        """
    )
    st.caption(
        "Attention: many RUL papers use **MSE on RUL head only**; extra terms are optional priors. "
        "PINN: $s$ sensors, $w$ wear proxy, $f_{\text{wear}}$ from physics literature."
    )

    st.markdown("#### Illustrative training dynamics (three families)")
    st.plotly_chart(fig_training_loss_curves(template=tpl), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(fig_radar_three_way(template=tpl), use_container_width=True)
    with r2:
        st.plotly_chart(fig_loss_components_bar(template=tpl), use_container_width=True)

    st.markdown("#### Intuition — convolution vs attention vs wear channel")
    st.plotly_chart(fig_sensor_window_physics_attention(template=tpl), use_container_width=True)

    mm_theme = "neutral" if st.session_state.get("ui_theme", "dark") == "light" else "dark"
    # Subgraph IDs must not clash with link sources; avoid en-dash in labels.
    mermaid = """
flowchart LR
    subgraph sg_cnn["1D CNN"]
        W1[Windows] --> C1[Conv]
        C1 --> H1[RUL head]
    end
    subgraph sg_tr["Transformer"]
        T1[Tokenize] --> A1[Attention]
        A1 --> H2[RUL head]
    end
    subgraph sg_pinn["PINN"]
        S1[Sensors] --> N1[Network]
        N1 --> R1[Wear residual]
        R1 --> L1[Total loss]
        S1 --> L1
    end
    H1 --> FD["C MAPSS FD001 to FD004"]
    H2 --> FD
    L1 --> FD
    """
    st.markdown("#### Architecture flow (qualitative)")
    components.html(
        f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head><body>
<div class="mermaid">{mermaid}</div>
<script>mermaid.initialize({{startOnLoad:true, theme:'{mm_theme}'}});</script>
</body></html>
        """,
        height=340,
    )

    with st.expander("Implementation checklist (for your assignment code)", expanded=False):
        st.markdown(
            r"""
- **CNN:** per-engine normalization → sliding windows → 1D conv backbone → RUL regression; RMSE vs `RUL_FD00x.txt`.  
- **Transformer:** stack **L** cycles as a sequence (or patch embeddings); encoder-only or encoder–decoder; watch **KV cache** length on edge.  
- **PINN-style:** supervised term + **wear / residual**; tune $\lambda$ on a validation engine set.  
- **Fairness:** same splits/seeds; compare **parameter count** or **latency** fairly (see **Compute** tab).
            """
        )


def tab_compute():
    st.subheader("Compute & deployment reality")
    st.markdown(
        """
**GEMM bottleneck:** In deep networks, most time is spent in **general matrix multiply (GEMM)** ops. A common FLOP heuristic for forward pass is on the order of
**~2 × parameters × tokens** (per layer block, architecture-dependent)—use it to explain **why wide, deep PHM nets are expensive**, not as an accounting standard.

**Memory (attention / history):** For attention-style models, **KV cache** size scales with **history length**—on-wing **edge** devices can hit RAM ceilings for long traces.

**Quantization:** **INT8 / INT4** weights/activations often yield **~3–8×** throughput gains for small accuracy loss—reasonable for **health monitoring** after validation.
        """
    )

    st.markdown("#### FLOP sanity check (order-of-magnitude)")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_params_m = st.number_input("Parameters (millions)", min_value=1.0, max_value=500.0, value=12.0, step=0.5)
    with c2:
        tokens = st.number_input("Tokens / forward (effective)", min_value=1, max_value=65536, value=512, step=64)
    with c3:
        layers = st.number_input("Depth reference (for intuition only)", min_value=1, max_value=200, value=24)

    params = n_params_m * 1e6
    flops_forward = 2 * params * float(tokens)
    st.metric(
        "Rough forward FLOPs (2 × P × T)",
        f"{flops_forward:.3e}",
        help="Illustrative; real kernels depend on fusion, sparsity, and architecture.",
    )
    st.caption(
        f"At ~{layers} layers (context only), emphasize that **GEMM-heavy** stacks dominate wall-clock—not the Streamlit UI."
    )

    st.markdown("#### Cost calculator — API vs self-hosted (200 aircraft)")
    ac = st.slider("Fleet size (aircraft)", 10, 500, 200)
    inf_per_ac_day = st.number_input("Inferences per aircraft per day", min_value=1, max_value=10000, value=48)
    api_per_1k = st.number_input("Managed API: $ / 1000 inferences", min_value=0.001, max_value=50.0, value=0.25, format="%.4f")
    monthly_fixed_host = st.number_input("Self-host: fixed $ / month (VM + support)", min_value=0.0, max_value=500000.0, value=3500.0)
    marginal_per_million = st.number_input("Self-host: extra $ / 1M extra inferences (electricity, burst)", min_value=0.0, max_value=500.0, value=12.0)

    n_inf = ac * inf_per_ac_day * 30
    api_cost = (n_inf / 1000.0) * api_per_1k
    self_cost = monthly_fixed_host + (n_inf / 1e6) * marginal_per_million

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Monthly inferences (approx.)", f"{n_inf:,.0f}")
    cc2.metric("API-style monthly $", f"${api_cost:,.2f}")
    cc3.metric("Self-hosted monthly $ (model)", f"${self_cost:,.2f}")

    st.info(
        "Treat this as a **communication aid**: real TCO includes MLOps, redundancy, certification, and on-call—"
        "the point is **money scales with inference volume and model size**, not the Streamlit UI."
    )


def main():
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "dark"

    default_root = resolve_cmapss_root()
    with st.sidebar:
        st.radio(
            "Appearance",
            ["Dark", "Light"],
            horizontal=True,
            key="theme_ap",
            index=0 if st.session_state.ui_theme == "dark" else 1,
        )
        st.session_state.ui_theme = "dark" if st.session_state.theme_ap == "Dark" else "light"
        st.caption("NASA C-MAPSS · turbofan PHM")
        st.divider()
        with st.expander("Authors & institution", expanded=False):
            c_a, c_b = st.columns(2)
            with c_a:
                if os.path.isfile(LOGO_BEIHANG):
                    _logo_on_white_tile(LOGO_BEIHANG, max_height_px=120)
            with c_b:
                if os.path.isfile(LOGO_RCSSTEAP):
                    _logo_on_white_tile(LOGO_RCSSTEAP, max_height_px=120)
            st.markdown(
                """
**Beihang University** (北京航空航天大学)  
*Beijing University of Aeronautics and Astronautics*

**RCSSTEAP (China)** — Regional Centre for Space Science and Technology Education in Asia and the Pacific

---

**Contributors**

| Name | Student ID |
|------|------------|
| Kodie Amo Kwame | LS2525226 |
| Sumara Alfred Salifu | LS2525245 |
| Peta Mimi Precious | LS2525255 |
                """
            )
        st.header("Data source")
        root = st.text_input(
            "Path to CMAPSSData folder",
            value=default_root,
            help="Folder containing train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt",
        ).strip()
        available_fd = list_available_datasets(root)
        st.caption(
            "Only datasets with **train_*.txt** or **test_*.txt** in this folder are listed. "
            "On **Streamlit Cloud**, you must **commit** those files into the repo `data/` folder."
        )
        if available_fd:
            fd = st.selectbox("Dataset", available_fd, index=0)
            train_df, test_df, rul, train_path, test_path, rul_path = _load_cmapss_bundle(root, fd)
            if train_df is None and test_df is None:
                st.error("Files disappeared or could not be read. Check permissions and paths.")
                test_df = load_test_fd001(default_local_test_path())
                train_df = None
                rul = None
                data_root_label = os.path.dirname(default_local_test_path())
            else:
                data_root_label = root
        else:
            st.warning(
                "**No NASA files** found in that folder (no `train_FD00x.txt` / `test_FD00x.txt`). "
                "Copy your CMAPSS **CMAPSSData** files here, or on GitHub add them under **`data/`** and redeploy."
            )
            fd = "FD001"
            train_df, test_df, rul = None, None, None
            train_path = cmapss_file(root, "train", fd)
            test_path = cmapss_file(root, "test", fd)
            rul_path = rul_file(root, fd)
            st.info("Showing **synthetic FD001-style** demo data until real files are available.")
            test_df = load_test_fd001(default_local_test_path())
            data_root_label = os.path.dirname(default_local_test_path())

        split = st.radio("Fleet trajectories use", ["test", "train"], horizontal=True, index=0)
        df_fleet = test_df if split == "test" else train_df
        if df_fleet is None:
            df_fleet = test_df if test_df is not None else train_df

        st.divider()
        st.markdown("**Install:** `pip install -r requirements.txt`  \\n**Run:** `streamlit run app.py`")
        st.caption("Override path with env `CMAPSS_DATA_DIR` if you prefer.")
        st.divider()
        st.metric("Rows (fleet table)", f"{len(df_fleet):,}")
        st.metric("Engines (fleet table)", f"{df_fleet['unit'].nunique()}")

    inject_engineering_theme(st.session_state.ui_theme)
    pillar_intro()

    section_downloads(data_root_label, fd, train_df, test_df, rul)

    cap_parts = [
        f"Dataset **{fd}** | Fleet view: **{split}**",
        f"Train file: `{train_path}` — {'ok' if train_df is not None else 'missing'}",
        f"Test file: `{test_path}` — {'ok' if test_df is not None else 'missing'}",
        f"RUL file: `{rul_path}` — {'ok' if rul is not None else 'missing'}",
    ]
    data_caption = " | ".join(cap_parts)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Fleet overview",
            "Exploratory analysis",
            "Data audit report",
            "Architecture selection",
            "Deep learning lab",
            "Compute & deployment",
        ]
    )
    with tab1:
        tab_fleet_overview(df_fleet, data_caption)
    with tab2:
        tab_eda(fd, train_df, test_df, rul)
    with tab3:
        tab_data_audit()
    with tab4:
        tab_architecture()
    with tab5:
        tab_cnn_pinn_lab()
    with tab6:
        tab_compute()


if __name__ == "__main__":
    main()
