from __future__ import annotations

import csv
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_openai import ChatOpenAI

from .config import CFG, ensure_dirs
from .qa import answer_question


RefAnswerFn = Callable[
    ...,
    Union[
        Tuple[str, List[Dict[str, Any]]],
        Tuple[str, List[Dict[str, Any]], Dict[str, Any]],
    ],
]


@dataclass
class EvalItem:
    id: str
    category: str
    question: str
    expected_keywords: List[str]
    min_citations: int
    should_refuse: bool


def load_evalset(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Evalset file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in evalset: {e}") from e

    if not isinstance(raw, list):
        raise ValueError("Evalset JSON must be a list of objects.")

    if not raw:
        raise ValueError("Evalset is empty.")

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Eval item at index {i} must be an object.")
        if not item.get("question"):
            raise ValueError(f"Eval item at index {i} is missing 'question'.")

    return raw


def _normalise_item(item: Dict, idx: int) -> EvalItem:
    item_id = item.get("id") or f"Q{idx + 1}"
    category = item.get("category")
    should_refuse = item.get("should_refuse")

    if category is None:
        category = "should_refuse" if should_refuse else "answerable"

    if should_refuse is None:
        should_refuse = category == "should_refuse"

    expected_keywords = item.get("expected_keywords") or []
    if not isinstance(expected_keywords, list):
        expected_keywords = [str(expected_keywords)]

    min_citations = item.get("min_citations")
    if min_citations is None:
        min_citations = 0 if should_refuse else 1

    return EvalItem(
        id=str(item_id),
        category=str(category),
        question=str(item.get("question", "")),
        expected_keywords=[str(k) for k in expected_keywords],
        min_citations=int(min_citations),
        should_refuse=bool(should_refuse),
    )


def _keyword_hit_rate(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    a = (answer or "").lower()
    hits = sum(1 for k in keywords if k.lower() in a)
    return hits / len(keywords)


def _is_refusal(answer: str, debug_refused: Optional[bool]) -> bool:
    if debug_refused is True:
        return True
    if answer is None:
        return True
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
        "could not find this information in the uploaded pdfs",
    ]
    return any(p in a for p in phrases)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(math.ceil((pct / 100.0) * len(values_sorted))) - 1
    k = max(0, min(k, len(values_sorted) - 1))
    return values_sorted[k]


def _truncate(text: str, max_chars: int = 1200) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "..."


def _judge_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CFG.CHAT_MODEL, temperature=0)


def _judge_metrics(question: str, answer: str, contexts: List[str]) -> Dict:
    if not contexts:
        return {
            "groundedness": {"score": 0.0, "rationale": "No retrieved context provided."},
            "context_relevance": {"score": 0.0, "rationale": "No retrieved context provided."},
            "answer_relevance": {"score": 0.0, "rationale": "No retrieved context provided."},
        }

    joined_context = "\n\n---\n\n".join(_truncate(c, 800) for c in contexts if c)
    system = (
        "You are a strict evaluator. Score each metric 0.0 to 1.0 and provide a short rationale. "
        "Return ONLY valid JSON."
    )
    user = {
        "question": question,
        "answer": answer,
        "contexts": joined_context,
        "instructions": (
            "Provide JSON with this exact shape:\n"
            "{\n"
            '  "groundedness": {"score": 0.0-1.0, "rationale": "..."},\n'
            '  "context_relevance": {"score": 0.0-1.0, "rationale": "..."},\n'
            '  "answer_relevance": {"score": 0.0-1.0, "rationale": "..."}\n'
            "}"
        ),
    }

    resp = _judge_llm().invoke([("system", system), ("user", json.dumps(user))])
    text = (resp.content or "").strip()

    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        return {
            "groundedness": {"score": 0.0, "rationale": "Judge output could not be parsed."},
            "context_relevance": {"score": 0.0, "rationale": "Judge output could not be parsed."},
            "answer_relevance": {"score": 0.0, "rationale": "Judge output could not be parsed."},
        }


def _call_with_timeout(fn: Callable[[], Any], timeout_s: int) -> Any:
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn)
        return future.result(timeout=timeout_s)


def run_evaluation(
    eval_items: List[Dict],
    *,
    judge_enabled: bool = False,
    timeout_s: int = 60,
    answer_fn: RefAnswerFn = answer_question,
    evalset_path: Optional[Path] = None,
) -> Dict:
    import openai as _openai
    try:
        _openai.models.list()
    except _openai.AuthenticationError:
        print("ERROR: OpenAI API key is invalid or missing. Cannot run evaluation.")
        return {"error": "invalid_api_key"}
    except Exception as e:
        print(f"WARNING: Could not verify API key before eval: {e}")

    ensure_dirs()

    items_norm = [_normalise_item(item, i) for i, item in enumerate(eval_items)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_items: List[Dict] = []
    latencies: List[float] = []
    keyword_rates: List[float] = []
    citation_flags: List[int] = []
    refusal_flags: List[int] = []

    judge_grounded: List[float] = []
    judge_context_rel: List[float] = []
    judge_answer_rel: List[float] = []

    for item in items_norm:
        t0 = time.perf_counter()
        error: Optional[str] = None

        def _call_answer():
            try:
                return answer_fn(item.question, return_debug=True)
            except TypeError:
                return answer_fn(item.question)

        try:
            result = _call_with_timeout(_call_answer, timeout_s)
            if isinstance(result, tuple) and len(result) == 3:
                answer_text, sources, debug = result
            else:
                answer_text, sources = result  # type: ignore[misc]
                debug = {}
        except TimeoutError:
            answer_text, sources, debug = "", [], {}
            error = f"timeout_after_{timeout_s}s"
        except Exception as e:
            answer_text, sources, debug = "", [], {}
            error = f"exception: {e}"

        latency = time.perf_counter() - t0
        latencies.append(latency)

        debug_refused = debug.get("refused") if isinstance(debug, dict) else None
        refused = _is_refusal(answer_text, debug_refused)

        contexts = []
        if isinstance(debug, dict):
            contexts = debug.get("retrieved_contexts") or []

        sources_used = [
            {
                "doc": s.get("source"),
                "page": s.get("page"),
                "chunk_id": s.get("chunk"),
            }
            for s in sources
        ]

        if isinstance(debug, dict) and debug.get("retrieval_debug"):
            retrieval_debug = debug.get("retrieval_debug")
        else:
            retrieval_debug = [
                {
                    "doc": s.get("source"),
                    "page": s.get("page"),
                    "chunk_id": s.get("chunk"),
                    "distance": s.get("distance"),
                }
                for s in sources
            ]

        citation_ok = len(sources) >= item.min_citations
        keyword_rate = _keyword_hit_rate(answer_text, item.expected_keywords)
        refusal_correct = refused == item.should_refuse

        keyword_rates.append(keyword_rate)
        citation_flags.append(1 if citation_ok else 0)
        refusal_flags.append(1 if refusal_correct else 0)

        judge_block = None
        if judge_enabled and error is None:
            judge_block = _judge_metrics(item.question, answer_text, contexts)
            try:
                judge_grounded.append(float(judge_block["groundedness"]["score"]))
                judge_context_rel.append(float(judge_block["context_relevance"]["score"]))
                judge_answer_rel.append(float(judge_block["answer_relevance"]["score"]))
            except Exception:
                pass

        failed = (
            (not refusal_correct)
            or (item.min_citations > 0 and not citation_ok)
            or (item.expected_keywords and keyword_rate < 1.0)
        )

        results_items.append(
            {
                "id": item.id,
                "category": item.category,
                "question": item.question,
                "answer": answer_text,
                "refused": refused,
                "sources_used": sources_used,
                "retrieval_debug": retrieval_debug,
                "retrieved_contexts": contexts,
                "latency_seconds": round(latency, 3),
                "metrics": {
                    "keyword_hit_rate": round(keyword_rate, 3),
                    "citation_ok": citation_ok,
                    "refusal_correct": refusal_correct,
                },
                "judge": judge_block,
                "error": error,
                "failed": failed,
            }
        )

    summary = {
        "n": len(results_items),
        "refusal_accuracy": round(sum(refusal_flags) / max(1, len(refusal_flags)), 3),
        "citation_coverage_rate": round(sum(citation_flags) / max(1, len(citation_flags)), 3),
        "avg_keyword_hit_rate": round(sum(keyword_rates) / max(1, len(keyword_rates)), 3),
        "median_latency": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95_latency": round(_percentile(latencies, 95), 3),
    }

    if judge_enabled:
        summary.update(
            {
                "avg_groundedness": round(sum(judge_grounded) / max(1, len(judge_grounded)), 3),
                "avg_context_relevance": round(sum(judge_context_rel) / max(1, len(judge_context_rel)), 3),
                "avg_answer_relevance": round(sum(judge_answer_rel) / max(1, len(judge_answer_rel)), 3),
            }
        )

    # Breakdown by category
    by_category: Dict[str, Dict] = {}
    for item in results_items:
        cat = item.get("category") or "uncategorised"
        bucket = by_category.setdefault(
            cat,
            {
                "category": cat,
                "n": 0,
                "refusal_accuracy": [],
                "citation_coverage_rate": [],
                "keyword_hit_rate": [],
                "latencies": [],
            },
        )
        bucket["n"] += 1
        bucket["refusal_accuracy"].append(1 if item["metrics"]["refusal_correct"] else 0)
        bucket["citation_coverage_rate"].append(1 if item["metrics"]["citation_ok"] else 0)
        bucket["keyword_hit_rate"].append(float(item["metrics"]["keyword_hit_rate"]))
        bucket["latencies"].append(float(item["latency_seconds"]))

    category_rows = []
    for cat, bucket in by_category.items():
        category_rows.append(
            {
                "category": cat,
                "n": bucket["n"],
                "refusal_accuracy": round(sum(bucket["refusal_accuracy"]) / max(1, bucket["n"]), 3),
                "citation_coverage_rate": round(sum(bucket["citation_coverage_rate"]) / max(1, bucket["n"]), 3),
                "avg_keyword_hit_rate": round(sum(bucket["keyword_hit_rate"]) / max(1, bucket["n"]), 3),
                "median_latency": round(statistics.median(bucket["latencies"]), 3)
                if bucket["latencies"]
                else 0.0,
            }
        )

    meta = {
        "timestamp": timestamp,
        "model": CFG.CHAT_MODEL,
        "retriever_k": CFG.TOP_K,
        "chunk_size": CFG.CHUNK_SIZE,
        "chunk_overlap": CFG.CHUNK_OVERLAP,
        "max_distance": getattr(CFG, "MAX_DISTANCE", None),
        "judge_enabled": judge_enabled,
        "timeout_s": timeout_s,
        "evalset_path": str(evalset_path) if evalset_path else None,
    }

    results = {
        "meta": meta,
        "summary": summary,
        "by_category": category_rows,
        "items": results_items,
    }

    # Write artifacts
    out_dir = Path(CFG.EVALSETS_DIR)
    json_path = out_dir / f"results_{timestamp}.json"
    csv_path = out_dir / f"results_{timestamp}.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "category",
            "question",
            "answer",
            "refused",
            "latency_seconds",
            "keyword_hit_rate",
            "citation_ok",
            "refusal_correct",
            "error",
        ]
        if judge_enabled:
            fieldnames += [
                "groundedness_score",
                "context_relevance_score",
                "answer_relevance_score",
            ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results_items:
            row = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": item["answer"],
                "refused": item["refused"],
                "latency_seconds": item["latency_seconds"],
                "keyword_hit_rate": item["metrics"]["keyword_hit_rate"],
                "citation_ok": item["metrics"]["citation_ok"],
                "refusal_correct": item["metrics"]["refusal_correct"],
                "error": item.get("error"),
            }
            if judge_enabled and item.get("judge"):
                row.update(
                    {
                        "groundedness_score": item["judge"]["groundedness"]["score"],
                        "context_relevance_score": item["judge"]["context_relevance"]["score"],
                        "answer_relevance_score": item["judge"]["answer_relevance"]["score"],
                    }
                )
            writer.writerow(row)

    results["artifacts"] = {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
    return results


def run_chat_history_evaluation(
    questions: List[str],
    *,
    allowed_sources: Optional[List[str]] = None,
    timeout_s: int = 60,
    answer_fn: RefAnswerFn = answer_question,
) -> Dict:
    ensure_dirs()
    results_items: List[Dict] = []
    latencies: List[float] = []
    refusal_flags: List[int] = []
    citation_flags: List[int] = []

    for q in questions:
        t0 = time.perf_counter()
        error: Optional[str] = None

        def _call_answer():
            try:
                return answer_fn(q, allowed_sources=allowed_sources, return_debug=True)
            except TypeError:
                return answer_fn(q, allowed_sources=allowed_sources)

        try:
            result = _call_with_timeout(_call_answer, timeout_s)
            if isinstance(result, tuple) and len(result) == 3:
                answer_text, sources, debug = result
            else:
                answer_text, sources = result  # type: ignore[misc]
                debug = {}
        except TimeoutError:
            answer_text, sources, debug = "", [], {}
            error = f"timeout_after_{timeout_s}s"
        except Exception as e:
            answer_text, sources, debug = "", [], {}
            error = f"exception: {e}"

        latency = time.perf_counter() - t0
        latencies.append(latency)

        debug_refused = debug.get("refused") if isinstance(debug, dict) else None
        refused = _is_refusal(answer_text, debug_refused)

        sources_used = [
            {
                "doc": s.get("source"),
                "page": s.get("page"),
                "chunk_id": s.get("chunk"),
            }
            for s in sources
        ]

        citation_ok = len(sources) >= 1
        citation_flags.append(1 if citation_ok else 0)
        refusal_flags.append(1 if refused else 0)

        results_items.append(
            {
                "question": q,
                "answer": answer_text,
                "refused": refused,
                "sources_used": sources_used,
                "latency_seconds": round(latency, 3),
                "citation_ok": citation_ok,
                "error": error,
            }
        )

    summary = {
        "n": len(results_items),
        "citation_coverage_rate": round(sum(citation_flags) / max(1, len(citation_flags)), 3),
        "refusal_rate": round(sum(refusal_flags) / max(1, len(refusal_flags)), 3),
        "refusal_count": int(sum(refusal_flags)),
        "median_latency": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95_latency": round(_percentile(latencies, 95), 3),
    }

    return {"summary": summary, "items": results_items}
