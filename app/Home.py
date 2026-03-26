import sys
from pathlib import Path

# ✅ Make sure the project root is on Python's import path
# This fixes: ModuleNotFoundError: No module named 'rag'
ROOT = Path(__file__).resolve().parents[1]  # .../kwasis-rag
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from rag.config import ensure_dirs, assert_api_key

st.set_page_config(
    page_title="Kwasi's RAG App",
    layout="wide",
)

# ✅ Ensure folders exist + API key exists
ensure_dirs()
assert_api_key()

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
