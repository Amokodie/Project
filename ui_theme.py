"""Minimal aerospace / turbofan accents + Dark/Light CSS (content-first)."""

from __future__ import annotations

import streamlit as st


def _svg_turbofan_corner(for_light: bool) -> str:
    """
    Very subtle side-view turbofan hint: fan disk + shaft + compressor ticks.
    Low opacity so charts and text stay the focus.
    """
    stroke = "%2394a3b8" if for_light else "%2338bdf8"
    op = "0.045" if for_light else "0.035"
    # Compact SVG — bottom-right placement via CSS; single group
    return (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' width='420' height='320' viewBox='0 0 420 320'%3E"
        f"%3Cg fill='none' stroke='{stroke}' stroke-opacity='{op}' stroke-width='1'%3E"
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
    mode: 'dark' | 'light'
    Keeps background minimal — one faint engine motif, no heavy gradients.
    """
    light = mode == "light"
    svg = _svg_turbofan_corner(light)

    if light:
        bg = "#f4f6f8"
        app_css = f"""
  .stApp {{
    background-color: {bg};
    background-image:
      url("data:image/svg+xml,{svg}");
    background-repeat: no-repeat;
    background-position: right -40px bottom -30px;
    background-size: 380px auto;
    background-attachment: fixed;
  }}
  [data-testid="stHeader"] {{
    background: rgba(255, 255, 255, 0.92);
    border-bottom: 1px solid rgba(15, 23, 42, 0.08);
  }}
  section[data-testid="stSidebar"] {{
    background: #f8fafc !important;
    border-right: 1px solid rgba(15, 23, 42, 0.08);
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
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(255, 255, 255, 0.9);
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid rgba(15, 23, 42, 0.08);
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid rgba(15, 23, 42, 0.1);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.7);
  }}
        """
    else:
        app_css = f"""
  .stApp {{
    background-color: #0c1118;
    background-image:
      url("data:image/svg+xml,{svg}");
    background-repeat: no-repeat;
    background-position: right -50px bottom -40px;
    background-size: 400px auto;
    background-attachment: fixed;
  }}
  [data-testid="stHeader"] {{
    background: rgba(12, 17, 24, 0.94);
    border-bottom: 1px solid rgba(56, 189, 248, 0.1);
  }}
  section[data-testid="stSidebar"] {{
    background: #0f1419 !important;
    border-right: 1px solid rgba(56, 189, 248, 0.1);
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
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(15, 23, 42, 0.35);
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid rgba(56, 189, 248, 0.1);
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.25);
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
    """Thin accent under title — toned down in light mode."""
    if mode == "light":
        grad = "linear-gradient(90deg, transparent 0%, #64748b 35%, #94a3b8 50%, #64748b 65%, transparent 100%)"
        op = "0.55"
    else:
        grad = "linear-gradient(90deg, transparent 0%, #0ea5e9 25%, #38bdf8 50%, #22d3ee 75%, transparent 100%)"
        op = "0.75"
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
