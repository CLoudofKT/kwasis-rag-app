import time
import streamlit as st
from pathlib import Path

from rag.config import CFG, ensure_dirs
from rag.ingest import save_upload, ingest_pdf, delete_pdf_from_memory
from rag.store import clear_vectorstore

# --- Page header ---
st.title("1) Upload PDFs")
st.write("Upload one or more PDFs. The app will index them so you can ask questions later.")

# Make sure required folders exist
ensure_dirs()
uploads_dir = Path(CFG.UPLOADS_DIR)

# --- Upload / ingest section ---
uploads = st.file_uploader(
    "Choose files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploads:
    for f in uploads:
        progress = st.progress(0, text=f"Uploading {f.name}...")
        t0 = time.perf_counter()
        saved_path = save_upload(f.getvalue(), f.name)
        progress.progress(40, text=f"Extracting + chunking {f.name}...")
        result = ingest_pdf(saved_path)
        progress.progress(100, text=f"Finished {f.name}")
        elapsed = time.perf_counter() - t0

        if result.get("warning"):
            st.warning(
                f"{result['file']}: {result['warning']}"
            )
        if result.get("chunks", 0) > 0:
            st.success(
                f"Indexed: {result['file']} | Pages: {result['pages']} | "
                f"Chunks: {result['chunks']} | Time: {elapsed:.2f}s"
            )
        else:
            st.info(
                f"Indexed: {result['file']} | Pages: {result['pages']} | "
                f"Chunks: {result['chunks']} | Time: {elapsed:.2f}s"
            )

st.divider()

# --- Stored PDFs section ---
st.subheader("Currently stored PDFs")

pdfs = sorted([p.name for p in uploads_dir.glob("*.pdf")])

if not pdfs:
    st.info("No PDFs uploaded yet.")
else:
    for name in pdfs:
        c1, c2 = st.columns([0.8, 0.2])
        c1.write(f"• {name}")

        if c2.button("Delete", key=f"del_{name}"):
            with st.spinner("Deleting from uploads + memory..."):
                result = delete_pdf_from_memory(name)

            st.success(
                f"Deleted: {result['file']} | "
                f"File removed: {result['deleted_file']} | "
                f"Chunks removed: {result['deleted_chunks']}"
            )
            st.rerun()

st.divider()
st.subheader("Reset (danger zone)")
st.write("Clear all PDFs and reset the chat memory + vector index.")
confirm = st.checkbox("I understand this will delete all uploaded PDFs and clear the index.")
if st.button("Clear all PDFs + reset memory", disabled=not confirm):
    for p in uploads_dir.glob("*.pdf"):
        p.unlink(missing_ok=True)
    clear_vectorstore()
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.success("All PDFs removed, vector index cleared, and chat memory reset.")
    st.rerun()
