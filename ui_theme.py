"""Clean solid backgrounds for Dark/Light; transparent main stack so the app shell color shows."""

from __future__ import annotations

import streamlit as st

# Match .streamlit/config.toml backgroundColor for dark (avoids Streamlit vs custom mismatch).
_BG_DARK = "#0c1118"
_BG_LIGHT = "#f1f5f9"


def inject_engineering_theme(mode: str = "dark") -> None:
    """
    Streamlit paints inner containers from [theme]; we set the shell to a flat color and
    force main blocks transparent so the same fill is visible in Dark and Light.
    """
    light = mode == "light"

    common_transparent = """
  [data-testid="stAppViewContainer"] {
    background: transparent !important;
  }
  section[data-testid="stMain"] > div {
    background: transparent !important;
  }
  section[data-testid="stMain"] {
    background: transparent !important;
  }
  .stApp .main {
    background: transparent !important;
  }
  .main > div {
    background: transparent !important;
  }
  section[data-testid="stMain"] .block-container {
    background: transparent !important;
  }
  section[data-testid="stMain"] [class*="block-container"] {
    background: transparent !important;
  }
  section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
  }
  section[data-testid="stMain"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
  }
"""

    if light:
        bg = _BG_LIGHT
        fg = "#0f172a"
        fg_muted = "#334155"
        fg_soft = "#475569"
        app_css = f"""
  html {{
    color-scheme: light !important;
  }}
  html, body {{
    background-color: {bg} !important;
    color: {fg} !important;
  }}
  .stApp {{
    background: {bg} !important;
    background-color: {bg} !important;
    background-image: none !important;
    min-height: 100vh;
    color: {fg} !important;
    --text-color: {fg} !important;
    --primary-text-color: {fg} !important;
    --secondary-text-color: {fg_muted} !important;
    --disabled-text-color: #94a3b8 !important;
  }}
{common_transparent}
  [data-testid="stAppViewContainer"] {{
    color: {fg} !important;
  }}
  section[data-testid="stMain"] {{
    color: {fg} !important;
  }}
  .stApp .main {{
    color: {fg} !important;
  }}
  [data-testid="stHeader"] {{
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    color: {fg} !important;
  }}
  section[data-testid="stSidebar"] {{
    background: #f8fafc !important;
    border-right: 1px solid rgba(15, 23, 42, 0.1);
    color: {fg} !important;
    --text-color: {fg} !important;
    --primary-text-color: {fg} !important;
    --secondary-text-color: {fg_muted} !important;
  }}
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] li {{
    color: {fg} !important;
  }}
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] textarea {{
    color: {fg} !important;
    -webkit-text-fill-color: {fg} !important;
    background-color: #ffffff !important;
  }}
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] ol,
  [data-testid="stMarkdownContainer"] ul,
  [data-testid="stMarkdownContainer"] strong,
  [data-testid="stMarkdownContainer"] em {{
    color: {fg} !important;
  }}
  [data-testid="stMarkdownContainer"] pre code {{
    color: #f1f5f9 !important;
  }}
  section[data-testid="stSidebar"] [data-baseweb="radio"] label,
  section[data-testid="stSidebar"] [data-baseweb="radio"] div {{
    color: {fg} !important;
  }}
  section[data-testid="stSidebar"] [data-baseweb="select"] div[class] {{
    color: {fg} !important;
  }}
  .stApp a {{
    color: #0369a1 !important;
  }}
  [data-testid="stCaption"] p,
  [data-testid="stCaption"] {{
    color: {fg_soft} !important;
  }}
  h1 {{
    color: {fg} !important;
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
    color: {fg} !important;
  }}
  div[data-testid="stMetric"] p,
  div[data-testid="stMetric"] span {{
    color: {fg} !important;
  }}
  div[data-testid="stMetric"] [data-testid="stMarkdownContainer"] p {{
    color: {fg_muted} !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: #ffffff !important;
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid rgba(15, 23, 42, 0.12);
  }}
  [data-baseweb="tab"] {{
    color: {fg_muted} !important;
  }}
  [data-baseweb="tab"][aria-selected="true"] {{
    color: {fg} !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid rgba(15, 23, 42, 0.12);
    border-radius: 10px;
    background: #ffffff !important;
    color: {fg} !important;
  }}
  div[data-testid="stExpander"] summary {{
    color: {fg} !important;
  }}
        """
    else:
        bg = _BG_DARK
        app_css = f"""
  html {{
    color-scheme: dark !important;
  }}
  html, body {{
    background-color: {bg} !important;
    color: #e5e7eb !important;
  }}
  .stApp {{
    background: {bg} !important;
    background-color: {bg} !important;
    background-image: none !important;
    min-height: 100vh;
    color: #e5e7eb !important;
    --text-color: #e5e7eb !important;
    --primary-text-color: #f8fafc !important;
    --secondary-text-color: #94a3b8 !important;
  }}
{common_transparent}
  [data-testid="stHeader"] {{
    background: rgba(12, 17, 24, 0.95) !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
  }}
  section[data-testid="stSidebar"] {{
    background: #0f1419 !important;
    border-right: 1px solid rgba(56, 189, 248, 0.12);
    color: #e5e7eb !important;
    --text-color: #e5e7eb !important;
    --primary-text-color: #f8fafc !important;
    --secondary-text-color: #94a3b8 !important;
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.1rem;
  }}
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] li {{
    color: #e5e7eb !important;
  }}
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] strong {{
    color: #e5e7eb !important;
  }}
  [data-testid="stCaption"] p {{
    color: #94a3b8 !important;
  }}
  .stApp a {{
    color: #38bdf8 !important;
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
  [data-baseweb="tab"] {{
    color: #94a3b8 !important;
  }}
  [data-baseweb="tab"][aria-selected="true"] {{
    color: #f0f9ff !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.4) !important;
  }}
  div[data-testid="stExpander"] summary {{
    color: #e5e7eb !important;
  }}
        """

    style_block = f"<style>{app_css}</style>"
    if hasattr(st, "html"):
        st.html(style_block)
    else:
        st.markdown(style_block, unsafe_allow_html=True)


def hero_engineering_ribbon(mode: str = "dark") -> None:
    """Thin accent under title."""
    if mode == "light":
        grad = "linear-gradient(90deg, transparent 0%, #94a3b8 40%, #64748b 50%, #94a3b8 60%, transparent 100%)"
        op = "0.5"
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
