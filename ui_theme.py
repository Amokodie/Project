"""Aerospace / turbofan accents + Dark/Light. Main area must stay transparent so .stApp background shows."""

from __future__ import annotations

import streamlit as st


def _svg_turbofan_corner(for_light: bool) -> str:
    """Slightly visible engine motif — bottom-right."""
    stroke = "%23475569" if for_light else "%2338bdf8"
    op = "0.11" if for_light else "0.09"
    return (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' width='420' height='320' viewBox='0 0 420 320'%3E"
        f"%3Cg fill='none' stroke='{stroke}' stroke-opacity='{op}' stroke-width='1.2'%3E"
        "%3Cellipse cx='210' cy='160' rx='88' ry='88'/%3E"
        "%3Cellipse cx='210' cy='160' rx='24' ry='88'/%3E"
        "%3Cline x1='120' y1='160' x2='300' y2='160'/%3E"
        "%3Cline x1='290' y1='100' x2='290' y2='220'/%3E"
        "%3Cline x1='310' y1='115' x2='310' y2='205'/%3E"
        "%3Cline x1='330' y1='125' x2='330' y2='195'/%3E"
        "%3Cpath d='M 40 160 Q 80 120 120 160 Q 80 200 40 160'/%3E"
        "%3C/g%3E"
        "%3C/svg%3E"
    )


def inject_engineering_theme(mode: str = "dark") -> None:
    """
    Streamlit paints the MAIN column with a solid theme color — that hides .stApp background.
    We set .stApp to the full scene (gradient + motif) and force main blocks to transparent.
    """
    light = mode == "light"
    svg = _svg_turbofan_corner(light)

    if light:
        # Soft cool-gray page + faint blue wash + motif
        bg_core = "#eef1f5"
        grad = (
            "radial-gradient(ellipse 90% 55% at 50% -15%, rgba(59, 130, 246, 0.12) 0%, transparent 55%), "
            "linear-gradient(180deg, #f8fafc 0%, #eef1f5 45%, #e8edf3 100%)"
        )
        app_css = f"""
  .stApp {{
    background-color: {bg_core} !important;
    background-image: {grad}, url("data:image/svg+xml,{svg}") !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center top, right -36px bottom -28px !important;
    background-size: auto, 400px auto !important;
    background-attachment: fixed, fixed !important;
    min-height: 100vh;
  }}
  [data-testid="stAppViewContainer"] {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] > div {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] {{
    background: transparent !important;
  }}
  .stApp .main {{
    background: transparent !important;
  }}
  .main > div {{
    background: transparent !important;
  }}
  [data-testid="stHeader"] {{
    background: rgba(255, 255, 255, 0.88) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(15, 23, 42, 0.1);
  }}
  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border-right: 1px solid rgba(15, 23, 42, 0.1);
  }}
  h1 {{
    color: #0f172a !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
  }}
  h2, h3 {{
    color: #1e3a5f !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stMetric"] {{
    background: #ffffff !important;
    border: 1px solid rgba(15, 23, 42, 0.1);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(255, 255, 255, 0.95) !important;
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid rgba(15, 23, 42, 0.1);
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid rgba(15, 23, 42, 0.12);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.85) !important;
  }}
        """
    else:
        bg_core = "#0a0f18"
        grad = (
            "radial-gradient(ellipse 100% 70% at 50% -20%, rgba(14, 165, 233, 0.14) 0%, transparent 55%), "
            "radial-gradient(ellipse 60% 45% at 100% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 50%), "
            "linear-gradient(180deg, #060a10 0%, #0a0f18 40%, #0c121c 100%)"
        )
        app_css = f"""
  .stApp {{
    background-color: {bg_core} !important;
    background-image: {grad}, url("data:image/svg+xml,{svg}") !important;
    background-repeat: no-repeat, no-repeat !important;
    background-position: center top, right -40px bottom -36px !important;
    background-size: auto, 420px auto !important;
    background-attachment: fixed, fixed !important;
    min-height: 100vh;
  }}
  [data-testid="stAppViewContainer"] {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] > div {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] {{
    background: transparent !important;
  }}
  .stApp .main {{
    background: transparent !important;
  }}
  .main > div {{
    background: transparent !important;
  }}
  [data-testid="stHeader"] {{
    background: rgba(6, 10, 16, 0.92) !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
  }}
  section[data-testid="stSidebar"] {{
    background: linear-gradient(185deg, #0d1219 0%, #080c12 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.14);
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.1rem;
  }}
  h1 {{
    color: #f0f9ff !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
  }}
  h2, h3 {{
    color: #bae6fd !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stMetric"] {{
    background: rgba(15, 23, 42, 0.55) !important;
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(15, 23, 42, 0.55) !important;
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid rgba(56, 189, 248, 0.12);
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.4) !important;
  }}
        """

    st.markdown(
        f"""
<style>
{app_css}
</style>
        """,
        unsafe_allow_html=True,
    )


def hero_engineering_ribbon(mode: str = "dark") -> None:
    """Thin accent under title."""
    if mode == "light":
        grad = "linear-gradient(90deg, transparent 0%, #64748b 35%, #94a3b8 50%, #64748b 65%, transparent 100%)"
        op = "0.65"
    else:
        grad = "linear-gradient(90deg, transparent 0%, #0ea5e9 25%, #38bdf8 50%, #22d3ee 75%, transparent 100%)"
        op = "0.85"
    st.markdown(
        f"""
<div style="
  height:3px;
  border-radius:2px;
  margin: 0 0 0.85rem 0;
  background: {grad};
  opacity: {op};
"></div>
        """,
        unsafe_allow_html=True,
    )


def plotly_template(mode: str) -> str:
    return "plotly_white" if mode == "light" else "plotly_dark"
