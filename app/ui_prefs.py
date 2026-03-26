from __future__ import annotations

from typing import Any, Dict

import streamlit as st

DEFAULT_PREFS: Dict[str, Any] = {
    "accent": "Red",
    "verbosity": "Balanced",
    "eval_show_failed_only": False,
    "eval_highlight_slow": True,
    "eval_slow_threshold_s": 6.0,
    "eval_highlight_low_keyword": True,
    "eval_keyword_threshold": 0.5,
}

ACCENT_COLORS = {
    "Red": "#E86A6A",
    "Blue": "#4C8BF5",
    "Green": "#43B581",
    "Purple": "#9B7BFF",
    "Orange": "#F59E0B",
}


def ensure_prefs() -> Dict[str, Any]:
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = DEFAULT_PREFS.copy()
    return st.session_state["prefs"]


def get_pref(key: str) -> Any:
    prefs = ensure_prefs()
    return prefs.get(key, DEFAULT_PREFS.get(key))


def set_pref(key: str, value: Any) -> None:
    prefs = ensure_prefs()
    prefs[key] = value
    st.session_state["prefs"] = prefs


def reset_prefs() -> None:
    st.session_state["prefs"] = DEFAULT_PREFS.copy()


def apply_accent_css(accent_name: str) -> None:
    color = ACCENT_COLORS.get(accent_name, ACCENT_COLORS["Red"])
    css = f"""
<style>
:root {{
  --accent: {color};
}}
.stButton > button, .stDownloadButton > button {{
  border: 1px solid var(--accent) !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background-color: var(--accent) !important;
  color: #ffffff !important;
}}
div[data-testid="stExpander"] summary {{
  border-left: 3px solid var(--accent);
  padding-left: 0.5rem;
}}
div[data-baseweb="tab"] > button[aria-selected="true"] {{
  border-bottom: 2px solid var(--accent) !important;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
