import re
import streamlit as st
from pathlib import Path

from app.ui_prefs import apply_accent_css, ensure_prefs, get_pref
from rag.config import CFG
from rag.qa import answer_question

st.title("2) Chat (Ask Questions)")
st.write("Ask questions and follow up per document. Each tab keeps its own conversation history.")

# Build list of available PDFs from uploads folder
uploads_dir = Path(CFG.UPLOADS_DIR)
uploads_dir.mkdir(parents=True, exist_ok=True)
pdf_names = sorted([p.name for p in uploads_dir.glob("*.pdf")])

ensure_prefs()
apply_accent_css(get_pref("accent"))

HISTORY_TURNS = 6  # last N user/assistant turns to include


def _scope_key(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return safe or "all"


def _messages_key(scope: str) -> str:
    return f"messages__{scope}"


def _get_messages(scope: str) -> list:
    key = _messages_key(scope)
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def _render_messages(messages: list) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                with st.expander("Sources used"):
                    if not sources:
                        st.caption("No sources returned.")
                    else:
                        for s in sources:
                            source = s.get("source", "unknown")
                            page = s.get("page", "?")
                            chunk = s.get("chunk", "?")
                            dist = s.get("distance")
                            dist_str = f"{dist:.3f}" if isinstance(dist, float) else "?"
                            label = f"{source} (p.{page}, chunk {chunk}) | distance={dist_str}"
                            st.markdown(f"**{label}**")
                            if s.get("snippet"):
                                st.caption(s["snippet"])


tab_labels = ["All PDFs"] + pdf_names
tabs = st.tabs(tab_labels)

for idx, tab in enumerate(tabs):
    with tab:
        if idx == 0:
            scope_label = "All PDFs"
            allowed_sources = None
        else:
            scope_label = pdf_names[idx - 1]
            allowed_sources = [scope_label]

        scope_key = _scope_key(scope_label)
        messages = _get_messages(scope_key)

        if not pdf_names:
            st.info("No PDFs uploaded yet. Go to Upload first.")

        c1, c2 = st.columns([0.7, 0.3])
        c1.subheader(f"Chat: {scope_label}")
        if c2.button("Clear this chat", key=f"clear_{scope_key}"):
            st.session_state[_messages_key(scope_key)] = []
            st.rerun()

        _render_messages(messages)

        with st.form(key=f"form_{scope_key}", clear_on_submit=True):
            question = st.text_input(
                "Ask a question",
                placeholder="Type your question and press Send",
                key=f"q_{scope_key}",
            )
            submitted = st.form_submit_button("Send")

        if submitted and question.strip():
            history_for_prompt = messages[-(HISTORY_TURNS * 2) :]
            messages.append({"role": "user", "content": question.strip()})
            st.session_state["active_chat_scope_key"] = scope_key
            st.session_state["active_chat_scope_label"] = scope_label
            with st.spinner("Thinking..."):
                verbosity = get_pref("verbosity")
                if verbosity == "Concise":
                    style_hint = "Answer concisely in 3–6 bullet points."
                elif verbosity == "Detailed":
                    style_hint = "Answer in a structured way with headings and more explanation."
                else:
                    style_hint = None
                answer, sources = answer_question(
                    question.strip(),
                    allowed_sources=allowed_sources,
                    history=history_for_prompt,
                    style_hint=style_hint,
                )
            messages.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()
