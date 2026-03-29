from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"
    EVALSETS_DIR: Path = DATA_DIR / "evalsets"

    # Vector store
    CHROMA_COLLECTION: str = "kwasis_docs"

    # Retrieval / chunking
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 5

    # IMPORTANT: distance threshold for Chroma (lower is better)
    # If too strict, it will always say "no reliable answer".
    MAX_DISTANCE: float = float(os.getenv("MAX_DISTANCE", "0.92"))

    # Confidence checks (simple heuristics)
    # If total context is tiny, treat retrieval as weak.
    MIN_CONTEXT_CHARS: int = int(os.getenv("MIN_CONTEXT_CHARS", "200"))
    # Only run keyword-overlap checks if the question has at least this many keywords.
    MIN_KEYWORDS_FOR_CHECK: int = int(os.getenv("MIN_KEYWORDS_FOR_CHECK", "2"))

    # Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")


CFG = AppConfig()


def ensure_dirs() -> None:
    CFG.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    CFG.EVALSETS_DIR.mkdir(parents=True, exist_ok=True)


def assert_api_key() -> None:
    key = CFG.OPENAI_API_KEY
    # On Streamlit Cloud, secrets may not be injected as env vars yet — try st.secrets fallback.
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")
            if key:
                os.environ["OPENAI_API_KEY"] = key
        except Exception:
            pass
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "• Streamlit Cloud: go to App Settings → Secrets and add:\n"
            '  OPENAI_API_KEY = "sk-..."\n'
            "• Locally: add OPENAI_API_KEY=sk-... to your .env file."
        )
    if not str(key).startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY looks invalid (expected a key starting with 'sk-'). "
            "Please check your Streamlit Secrets or .env file and replace it with a valid key."
        )
    
