from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List

from .config import CFG
from .qa import answer_question


def load_evalset(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def keyword_score(answer: str, keywords: List[str]) -> float:
    """
    Simple proxy metric: % of expected keywords present in answer.
    """
    if not keywords:
        return 0.0
    a = (answer or "").lower()
    hits = sum(1 for k in keywords if k.lower() in a)
    return hits / len(keywords)


def is_refusal(answer: str) -> bool:
    """
    Heuristic: detect "don't know / not found" style responses.
    """
    a = (answer or "").lower()
    phrases = [
        "i could not find",
        "i couldn't find",
        "no relevant content",
        "not present in the uploaded",
        "i don't know",
        "i do not know",
        "unable to find",
        "couldn't find a reliable answer",
    ]
    return any(p in a for p in phrases)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", (text or "").lower())


def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i : i + n]) for i in range(0, max(len(tokens) - n + 1, 0))]


def faithfulness_score(answer: str, context: str) -> float:
    """
    Simple faithfulness proxy: % of 3-grams in answer that also appear in context.
    """
    a_tokens = _tokenize(answer)
    c_tokens = _tokenize(context)
    if len(a_tokens) < 3 or len(c_tokens) < 3:
        return 0.0

    a_ngrams = _ngrams(a_tokens, 3)
    c_ngrams = set(_ngrams(c_tokens, 3))
    if not a_ngrams:
        return 0.0

    hits = sum(1 for g in a_ngrams if g in c_ngrams)
    return hits / len(a_ngrams)


def run_eval(eval_items: List[Dict]) -> Dict:
    rows = []
    start_all = time.perf_counter()

    for item in eval_items:
        q = item["question"]
        expected = item.get("expected_keywords", [])
        expected_answer = item.get("expected_answer")
        should_refuse = item.get("should_refuse")
        if should_refuse is None:
            should_refuse = (not expected) and (not expected_answer)

        t0 = time.perf_counter()
        ans, sources = answer_question(q)
        latency = time.perf_counter() - t0

        citations = len(sources)
        citation_coverage = 1.0 if citations > 0 else 0.0
        refusal = is_refusal(ans)
        refusal_correct = 1.0 if (refusal == should_refuse) else 0.0
        context_text = " ".join(s.get("snippet", "") for s in sources)
        faithful = faithfulness_score(ans, context_text)

        rows.append(
            {
                "question": q,
                "latency_s": round(latency, 3),
                "keyword_score": round(keyword_score(ans, expected), 3),
                "citations": citations,
                "citation_coverage": round(citation_coverage, 3),
                "refusal_correctness": round(refusal_correct, 3),
                "faithfulness": round(faithful, 3),
            }
        )

    total = time.perf_counter() - start_all
    avg_latency = sum(r["latency_s"] for r in rows) / max(len(rows), 1)
    avg_score = sum(r["keyword_score"] for r in rows) / max(len(rows), 1)
    avg_citation_cov = sum(r["citation_coverage"] for r in rows) / max(len(rows), 1)
    avg_refusal = sum(r["refusal_correctness"] for r in rows) / max(len(rows), 1)
    avg_faithful = sum(r["faithfulness"] for r in rows) / max(len(rows), 1)

    return {
        "summary": {
            "n": len(rows),
            "total_time_s": round(total, 2),
            "avg_latency_s": round(avg_latency, 3),
            "avg_keyword_score": round(avg_score, 3),
            "avg_citation_coverage": round(avg_citation_cov, 3),
            "avg_refusal_correctness": round(avg_refusal, 3),
            "avg_faithfulness": round(avg_faithful, 3),
        },
        "rows": rows,
    }
