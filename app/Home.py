import sys
from pathlib import Path

# ✅ Make sure the project root is on Python's import path
# This fixes: ModuleNotFoundError: No module named 'rag'
ROOT = Path(__file__).resolve().parents[1]  # .../kwasis-rag
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from rag.config import CFG, ensure_dirs, assert_api_key

st.set_page_config(
    page_title="Kwasi's RAG App",
    layout="wide",
)

# ✅ Ensure folders exist + API key exists
ensure_dirs()
try:
    assert_api_key()
except RuntimeError as e:
    st.error(f"**OpenAI API Key Error:** {e}")
    st.stop()

# Auto-ingest preloaded PDFs once per session if the vector store is empty
# Place PDFs in data/preloaded/ to have them ingested automatically:
#   - 2015-16 NBA season - Wikipedia.pdf
#   - Joe & the Juice MENU.pdf
#   - EEE_6_PRO_Module Guide_25-26.pdf
if not st.session_state.get("_preloaded_checked"):
    st.session_state["_preloaded_checked"] = True
    try:
        from rag.store import get_vectorstore
        from rag.ingest import ingest_pdf
        _vs = get_vectorstore()
        if _vs._collection.count() == 0:
            _preloaded_dir = CFG.DATA_DIR / "preloaded"
            _pdfs = sorted(_preloaded_dir.glob("*.pdf")) if _preloaded_dir.exists() else []
            if _pdfs:
                with st.spinner("Loading documents..."):
                    _ingested = []
                    for _pdf_path in _pdfs:
                        _dest = CFG.UPLOADS_DIR / _pdf_path.name
                        if not _dest.exists():
                            _dest.write_bytes(_pdf_path.read_bytes())
                        _ingested.append(ingest_pdf(_pdf_path))
                _names = ", ".join(r["file"] for r in _ingested)
                st.success(f"Loaded {len(_ingested)} preloaded document(s): {_names}")
    except Exception as _e:
        st.warning(f"Auto-ingestion skipped: {_e}")

st.title("Kwasi’s Retrieval-Augmented Generation (RAG) App")
st.caption("Ask questions over your uploaded documents using Retrieval-Augmented Generation (RAG).")
st.caption("Answers include citations, and evaluation tools help you measure quality and performance.")

st.markdown(
    """
**Use the sidebar to:**
- 📤 **Upload**: add and index documents for retrieval
- 💬 **Chat**: ask questions and receive answers with citations
- 📊 **Evaluate**: run a labelled evalset or evaluate recent chat; view metrics + insights
- 📘 **About & Glossary**: how the system works + metric definitions
- ⚙️ **Settings**: customise UI preferences only (theme accent, verbosity, evaluation display)
"""
)

st.subheader("Recommended workflow")
st.markdown(
    """
1. Upload documents  
2. Ask questions in Chat  
3. Run Evaluate (evalset or recent chat)  
4. Use Evaluation insights to identify improvements
"""
)

st.info(
    "Evaluate supports two modes: (a) formal labelled evalsets for benchmarking, "
    "(b) recent chat diagnostics for quick feedback."
)
