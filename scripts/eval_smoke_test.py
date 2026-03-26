from __future__ import annotations

from pathlib import Path

from rag.eval_runner import run_evaluation


def _mock_answer(question: str, return_debug: bool = False, **kwargs):
    if "capital of france" in question.lower():
        answer = "I could not find this information in the uploaded PDFs."
        sources = []
        debug = {"refused": True, "retrieval_debug": [], "retrieved_contexts": []}
    else:
        answer = "The return policy allows a refund with a receipt."
        sources = [
            {
                "source": "policy.pdf",
                "page": 1,
                "chunk": 0,
                "distance": 0.12,
                "snippet": "return policy refund receipt",
            }
        ]
        debug = {
            "refused": False,
            "retrieval_debug": [{"doc": "policy.pdf", "page": 1, "chunk_id": 0, "distance": 0.12}],
            "retrieved_contexts": ["Return policy: refunds require a receipt."],
        }
    if return_debug:
        return answer, sources, debug
    return answer, sources


def main() -> None:
    eval_items = [
        {
            "id": "A1",
            "category": "answerable",
            "question": "What is the return policy?",
            "expected_keywords": ["return", "refund", "receipt"],
            "min_citations": 1,
            "should_refuse": False,
        },
        {
            "id": "B1",
            "category": "should_refuse",
            "question": "What is the capital of France?",
            "should_refuse": True,
            "min_citations": 0,
        },
    ]

    results = run_evaluation(
        eval_items,
        judge_enabled=False,
        timeout_s=5,
        answer_fn=_mock_answer,
        evalset_path=Path("data/evalsets/evalset.json"),
    )

    summary = results["summary"]
    assert summary["refusal_accuracy"] == 1.0, "Refusal accuracy should be 1.0"
    assert summary["citation_coverage_rate"] == 1.0, "Citation coverage should be 1.0"
    assert summary["avg_keyword_hit_rate"] == 1.0, "Keyword hit rate should be 1.0"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
