import streamlit as st

from app.ui_prefs import apply_accent_css, ensure_prefs, get_pref, reset_prefs, set_pref
from rag.config import CFG

st.title("Settings & Preferences")
st.caption(
    "Customise the interface and output format. Core system parameters are read-only."
)

ensure_prefs()
apply_accent_css(get_pref("accent"))

st.subheader("Appearance")
accent_options = ["Red", "Blue", "Green", "Purple", "Orange"]
accent = st.radio(
    "Accent colour",
    options=accent_options,
    index=accent_options.index(get_pref("accent")),
    horizontal=True,
    key="pref_accent",
)
set_pref("accent", accent)

st.subheader("Answer display")
verbosity_options = ["Concise", "Balanced", "Detailed"]
verbosity = st.radio(
    "Answer verbosity",
    options=verbosity_options,
    index=verbosity_options.index(get_pref("verbosity")),
    horizontal=True,
    key="pref_verbosity",
)
set_pref("verbosity", verbosity)

st.subheader("Evaluation display")
eval_show_failed_only = st.checkbox(
    "Show failed only by default",
    value=bool(get_pref("eval_show_failed_only")),
    key="pref_eval_show_failed_only",
)
set_pref("eval_show_failed_only", eval_show_failed_only)

eval_highlight_slow = st.checkbox(
    "Highlight slow responses",
    value=bool(get_pref("eval_highlight_slow")),
    key="pref_eval_highlight_slow",
)
set_pref("eval_highlight_slow", eval_highlight_slow)

if eval_highlight_slow:
    slow_threshold = st.number_input(
        "Slow threshold (seconds)",
        min_value=1.0,
        max_value=60.0,
        value=float(get_pref("eval_slow_threshold_s")),
        step=0.5,
        key="pref_eval_slow_threshold_s",
    )
    set_pref("eval_slow_threshold_s", slow_threshold)

eval_highlight_low_keyword = st.checkbox(
    "Highlight low keyword-hit answers",
    value=bool(get_pref("eval_highlight_low_keyword")),
    key="pref_eval_highlight_low_keyword",
)
set_pref("eval_highlight_low_keyword", eval_highlight_low_keyword)

if eval_highlight_low_keyword:
    keyword_threshold = st.number_input(
        "Keyword hit threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(get_pref("eval_keyword_threshold")),
        step=0.05,
        key="pref_eval_keyword_threshold",
    )
    set_pref("eval_keyword_threshold", keyword_threshold)

if st.button("Reset preferences"):
    reset_prefs()
    st.success("Preferences reset to defaults.")
    st.rerun()

st.subheader("System configuration (read-only)")
st.caption("These values control core system behaviour and cannot be edited here.")

st.code(
    f"""
UPLOADS_DIR:     {CFG.UPLOADS_DIR}
VECTORSTORE_DIR: {CFG.VECTORSTORE_DIR}
EVALSETS_DIR:    {CFG.EVALSETS_DIR}

CHUNK_SIZE:      {CFG.CHUNK_SIZE}
CHUNK_OVERLAP:   {CFG.CHUNK_OVERLAP}
TOP_K:           {CFG.TOP_K}

EMBEDDING_MODEL: {CFG.EMBEDDING_MODEL}
CHAT_MODEL:      {CFG.CHAT_MODEL}
""".strip(),
    language="text",
)

st.caption(
    "- `UPLOADS_DIR`: where your PDFs are stored.\n"
    "- `VECTORSTORE_DIR`: where embeddings are persisted.\n"
    "- `EVALSETS_DIR`: where evaluation sets and results are saved.\n"
    "- `CHUNK_SIZE` / `CHUNK_OVERLAP`: how documents are split for retrieval.\n"
    "- `TOP_K`: how many chunks are retrieved per question.\n"
    "- `EMBEDDING_MODEL` / `CHAT_MODEL`: models used for retrieval and answering."
)
