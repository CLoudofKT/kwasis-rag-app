from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .config import CFG, ensure_dirs
from .store import add_texts, delete_by_source


def save_upload(file_bytes: bytes, filename: str) -> Path:
    """
    Save an uploaded PDF into data/uploads and return the saved path.
    """
    ensure_dirs()
    safe_name = filename.replace("/", "_").replace("\\", "_")
    dest = CFG.UPLOADS_DIR / safe_name
    dest.write_bytes(file_bytes)
    return dest


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CFG.CHUNK_SIZE,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )


def _try_load_pdf_text(pdf_path: Path) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def chunk_documents(docs: List[Document]) -> Tuple[List[str], List[dict], List[str]]:
    splitter = _splitter()

    texts: List[str] = []
    metas: List[dict] = []
    ids: List[str] = []

    for d in docs:
        source = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page", None)

        content = (d.page_content or "").strip()
        if not content:
            continue

        for i, chunk in enumerate(splitter.split_text(content)):
            texts.append(chunk)
            meta = {
                "source": source,
                "page": page,
                "chunk": i,
            }
            metas.append(meta)

            raw = f"{source}|{page}|{i}|{chunk}".encode("utf-8")
            ids.append(hashlib.sha1(raw).hexdigest())

    return texts, metas, ids


def ingest_pdf(pdf_path: Path) -> dict:
    """
    Ingest a PDF into Chroma (minimal flow: PDF -> chunks -> embeddings).
    """
    docs = _try_load_pdf_text(pdf_path)
    texts, metas, ids = chunk_documents(docs)

    if not texts:
        return {
            "file": pdf_path.name,
            "pages": len(docs) if docs else 0,
            "chunks": 0,
            "warning": "No readable text found in this PDF.",
        }

    added = add_texts(texts=texts, metadatas=metas, ids=ids)
    return {"file": pdf_path.name, "pages": len(docs), "chunks": added}


def delete_pdf_from_memory(filename: str) -> dict:
    """
    Deletes a PDF from uploads folder AND removes its chunks from Chroma.
    """
    uploads_dir = Path(CFG.UPLOADS_DIR)
    target = uploads_dir / filename

    deleted_chunks = delete_by_source(filename)

    deleted_file = False
    if target.exists():
        target.unlink()
        deleted_file = True

    return {
        "file": filename,
        "deleted_file": deleted_file,
        "deleted_chunks": deleted_chunks,
    }
