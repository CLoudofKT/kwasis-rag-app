from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .config import CFG, assert_api_key, ensure_dirs


def _embeddings() -> OpenAIEmbeddings:
    assert_api_key()
    return OpenAIEmbeddings(model=CFG.EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    ensure_dirs()
    return Chroma(
        collection_name=CFG.CHROMA_COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=str(CFG.VECTORSTORE_DIR),
    )


def add_texts(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
) -> int:
    vs = get_vectorstore()
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(texts)


def similarity_search(query: str, k: Optional[int] = None) -> List[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(query=query, k=k or CFG.TOP_K)


def similarity_search_with_score(
    query: str,
    k: Optional[int] = None,
    sources: Optional[List[str]] = None,
) -> List[Tuple[Document, float]]:
    """
    Returns (Document, distance) pairs. Lower distance = better.
    Optionally filter by filenames stored in metadata["source"].
    """
    vs = get_vectorstore()

    where = None
    if sources:
        # Chroma filter format:
        where = {"source": {"$in": sources}}

    # Different LangChain versions use different param names:
    try:
        return vs.similarity_search_with_score(query=query, k=k or CFG.TOP_K, filter=where)
    except TypeError:
        return vs.similarity_search_with_score(query=query, k=k or CFG.TOP_K, where=where)


def delete_collection() -> None:
    vs = get_vectorstore()
    vs.delete_collection()


def clear_vectorstore() -> None:
    delete_collection()


def delete_by_source(source_filename: str) -> int:
    """
    Deletes all chunks where metadata.source == source_filename.
    Returns best-effort number deleted.
    """
    vs = get_vectorstore()
    try:
        before = vs._collection.count()
        vs._collection.delete(where={"source": source_filename})
        after = vs._collection.count()
        return max(0, before - after)
    except Exception:
        return 0
