"""Inject global CSS for an aerospace / PHM engineering look (dark + turbine accents)."""

from __future__ import annotations

import streamlit as st


def inject_engineering_theme() -> None:
    """Layered background, sidebar accent, typography — runs once per app view."""
    # Subtle SVG: abstract flow lines + ring (reads as engine / nacelle motif at low opacity)
    svg = (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='600' viewBox='0 0 600 600'%3E"
        "%3Cg fill='none' stroke='%2338bdf8' stroke-opacity='0.11' stroke-width='1.2'%3E"
        "%3Cellipse cx='300' cy='300' rx='220' ry='90' transform='rotate(-8 300 300)'/%3E"
        "%3Cellipse cx='300' cy='300' rx='180' ry='72' transform='rotate(-8 300 300)'/%3E"
        "%3Cpath d='M40 300 Q150 260 300 300 T560 300'/%3E"
        "%3Cpath d='M40 320 Q150 360 300 320 T560 320'/%3E"
        "%3Cpath d='M80 180 Q200 140 300 180 Q400 220 520 180'/%3E"
        "%3Cpath d='M80 420 Q200 460 300 420 Q400 380 520 420'/%3E"
        "%3C/g%3E"
        "%3Cg fill='%2338bdf8' fill-opacity='0.04'%3E"
        "%3Ccircle cx='120' cy='120' r='80'/%3E"
        "%3Ccircle cx='520' cy='480' r='100'/%3E"
        "%3C/g%3E"
        "%3C/svg%3E"
    )
    st.markdown(
        f"""
<style>
  .stApp {{
    background-color: #070b14;
    background-image:
      radial-gradient(ellipse 120% 80% at 15% -10%, rgba(56, 189, 248, 0.14) 0%, transparent 45%),
      radial-gradient(ellipse 90% 60% at 100% 30%, rgba(59, 130, 246, 0.1) 0%, transparent 42%),
      radial-gradient(ellipse 70% 50% at 50% 100%, rgba(14, 165, 233, 0.08) 0%, transparent 50%),
      url("data:image/svg+xml,{svg}");
    background-attachment: fixed;
    background-size: auto, auto, auto, 520px 520px;
    background-position: center top, center, center, center;
  }}
  [data-testid="stHeader"] {{
    background: rgba(7, 11, 20, 0.92);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
  }}
  section[data-testid="stSidebar"] {{
    background: linear-gradient(185deg, #0c1222 0%, #070b14 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.14);
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.25rem;
  }}
  h1 {{
    color: #f0f9ff !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    text-shadow: 0 0 40px rgba(56, 189, 248, 0.15);
  }}
  h2, h3 {{
    color: #bae6fd !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stMetric"] {{
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.75) 0%, rgba(7, 11, 20, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(15, 23, 42, 0.5);
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid rgba(56, 189, 248, 0.12);
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    font-weight: 600;
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.35);
  }}
</style>
        """,
        unsafe_allow_html=True,
    )


def hero_engineering_ribbon() -> None:
    """Thin accent bar under the title strip."""
    st.markdown(
        """
<div style="
  height:4px;
  border-radius:3px;
  margin: 0 0 1rem 0;
  background: linear-gradient(90deg, transparent 0%, #0ea5e9 25%, #38bdf8 50%, #22d3ee 75%, transparent 100%);
  opacity: 0.95;
"></div>
        """,
        unsafe_allow_html=True,
    )
