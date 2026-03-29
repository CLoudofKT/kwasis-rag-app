import csv
import html
import io
import json
import statistics
import time
from pathlib import Path

import streamlit as st

from app.ui_prefs import apply_accent_css, ensure_prefs, get_pref
from rag.config import CFG
from rag.eval_runner import load_evalset, run_evaluation, run_chat_history_evaluation

MODE_FORMAL = "Formal evaluation (evalset.json)"
MODE_CHAT = "Chat history diagnostics (unlabelled)"

METRIC_GLOSSARY = {
    "Refusal accuracy": {
        "hint": "Higher is better",
        "help": "Whether refusals happened only when they should. Higher is better.",
        "definition": "Checks if the model refused only on questions that should be refused.",
    },
    "Citation coverage": {
        "hint": "Higher is better",
        "help": "Share of questions that include at least one citation. Higher is better.",
        "definition": "How often answers include at least one cited source.",
    },
    "Avg keyword hit rate": {
        "hint": "Higher is better",
        "help": "Average fraction of expected keywords found in answers. Higher is better.",
        "definition": "On average, how many expected keywords appeared in answers.",
    },
    "Median latency (s)": {
        "hint": "Lower is better",
        "help": "Typical response time in seconds. Lower is better.",
        "definition": "The middle response time across all questions.",
    },
    "P95 latency (s)": {
        "hint": "Lower is better",
        "help": "Slow-case response time in seconds (95% are faster). Lower is better.",
        "definition": "A high-latency marker; 95% of answers are faster than this.",
    },
    "Refusal rate": {
        "hint": "Lower is better",
        "help": "Share of questions where the model refused. Lower is better.",
        "definition": "How often the model refused to answer in this run.",
    },
    "Refusal count": {
        "hint": "Lower is better",
        "help": "Number of refusals in this run. Lower is better.",
        "definition": "Total count of refusals.",
    },
    "Avg groundedness": {
        "hint": "Higher is better",
        "help": "LLM-judge estimate of how well answers are supported by context.",
        "definition": "LLM judge score for whether the answer is supported by the retrieved context.",
    },
    "Avg context relevance": {
        "hint": "Higher is better",
        "help": "LLM-judge estimate of whether retrieved context is relevant.",
        "definition": "LLM judge score for how relevant the retrieved context is to the question.",
    },
    "Avg answer relevance": {
        "hint": "Higher is better",
        "help": "LLM-judge estimate of how well the answer addresses the question.",
        "definition": "LLM judge score for whether the answer addresses the question.",
    },
    "citation_ok": {
        "definition": "Whether a single answer met the minimum citation requirement.",
    },
    "refusal_correct": {
        "definition": "Whether a single refusal was appropriate for that question.",
    },
    "keyword hit rate": {
        "definition": "Fraction of expected keywords present in one answer.",
    },
    "per-question latency": {
        "definition": "Time it took to answer one question.",
    },
    "LLM judge metrics": {
        "definition": "Scores from a model that checks groundedness and relevance.",
    },
}


def _metric_card(col, label, value):
    info = METRIC_GLOSSARY.get(label, {})
    help_text = info.get("help") or info.get("definition") or ""
    hint = info.get("hint", "")
    tooltip = html.escape(help_text)
    header = (
        f"<div style='display:flex;align-items:center;gap:6px;'>"
        f"<span>{label}</span>"
        f"<span title='{tooltip}' style='cursor:help;color:#999;'>ⓘ</span>"
        f"</div>"
    )
    col.markdown(header, unsafe_allow_html=True)
    col.markdown(
        f"<div style='font-size:28px;font-weight:600;line-height:1.2;'>{value}</div>",
        unsafe_allow_html=True,
    )
    if hint:
        col.caption(hint)


def _get_active_chat_messages():
    scope_key = st.session_state.get("active_chat_scope_key")
    scope_label = st.session_state.get("active_chat_scope_label")
    if not scope_key:
        return None, None, None
    messages = st.session_state.get(f"messages__{scope_key}", [])
    return scope_key, scope_label, messages


def _append_to_evalset(item, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    if not isinstance(data, list):
        data = []
    data.append(item)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_evalset_lookup(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    lookup = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        expected_keywords = item.get("expected_keywords") or []
        lookup[item_id] = expected_keywords
    return lookup


def _status_formal(item, expected_keywords_present):
    metrics = item.get("metrics", {})
    refusal_correct = metrics.get("refusal_correct")
    citation_ok = metrics.get("citation_ok")
    keyword_hit_rate = metrics.get("keyword_hit_rate")

    if refusal_correct is False:
        return "fail"
    if not citation_ok:
        return "review"
    if expected_keywords_present and (keyword_hit_rate is not None) and keyword_hit_rate < 0.5:
        return "review"
    return "pass"


def _status_chat(item):
    return "pass" if item.get("citation_ok") else "review"


def _percentile(values, pct):
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int((pct / 100.0) * (len(values_sorted) - 1))
    return values_sorted[max(0, min(k, len(values_sorted) - 1))]


def _build_insights(rows, slow_threshold, keyword_threshold, mode):
    insights = []
    recommendations = []

    if not rows:
        return ["No results available yet."], ["Run an evaluation to see insights."]

    total = len(rows)

    def _is_fail(row):
        status = row.get("status")
        if status:
            return status == "fail"
        if row.get("citation_ok") is False:
            return True
        if row.get("refusal_correct") is False:
            return True
        return False

    fail_count = sum(1 for r in rows if _is_fail(r))
    pass_count = total - fail_count
    insights.append(f"{pass_count}/{total} passed; {fail_count} need review.")

    citation_flags = [r.get("citation_ok") for r in rows if r.get("citation_ok") is not None]
    if citation_flags:
        cite_rate = sum(1 for v in citation_flags if v) / max(1, len(citation_flags))
        insights.append(f"Citation coverage: {cite_rate:.0%} of questions.")
        if cite_rate < 0.8:
            recommendations.append(
                "Increase citation emphasis or require at least one cited source."
            )

    refusal_flags = [r.get("refusal_correct") for r in rows if r.get("refusal_correct") is not None]
    if refusal_flags:
        refusal_rate = sum(1 for v in refusal_flags if v) / max(1, len(refusal_flags))
        insights.append(f"Refusal accuracy: {refusal_rate:.0%} of cases.")

    reasons = []
    missing_citations = sum(1 for r in rows if r.get("citation_ok") is False)
    if missing_citations:
        reasons.append(("Missing citations", missing_citations))
    refusal_incorrect = sum(1 for r in rows if r.get("refusal_correct") is False)
    if refusal_incorrect:
        reasons.append(("Refusal incorrect", refusal_incorrect))
    low_keyword = sum(
        1
        for r in rows
        if r.get("keyword_hit_rate") is not None
        and r.get("keyword_hit_rate") < keyword_threshold
    )
    if low_keyword:
        reasons.append(("Low keyword hit", low_keyword))

    latencies = [r.get("latency_seconds") for r in rows if r.get("latency_seconds") is not None]
    slow_count = sum(1 for v in latencies if v >= slow_threshold) if latencies else 0
    if slow_count:
        reasons.append(("Slow responses", slow_count))

    if reasons:
        reasons = sorted(reasons, key=lambda x: x[1], reverse=True)[:3]
        reasons_text = ", ".join([f"{name} ({count})" for name, count in reasons])
        insights.append(f"Top issues: {reasons_text}.")

    if latencies:
        med = statistics.median(latencies)
        p95 = _percentile(latencies, 95)
        insights.append(f"Latency median {med:.1f}s; P95 {p95:.1f}s.")
        if slow_count:
            insights.append(f"{slow_count}/{total} are slow (≥{slow_threshold:.1f}s).")
            slow_fail = sum(
                1
                for r in rows
                if r.get("latency_seconds") is not None
                and r.get("latency_seconds") >= slow_threshold
                and _is_fail(r)
            )
            fast_count = total - slow_count
            if fast_count > 0:
                fast_fail = sum(
                    1
                    for r in rows
                    if r.get("latency_seconds") is not None
                    and r.get("latency_seconds") < slow_threshold
                    and _is_fail(r)
                )
                if slow_fail / slow_count > fast_fail / max(1, fast_count):
                    insights.append("Slow questions fail more often than fast ones.")

    if any(r.get("keyword_hit_rate") is not None for r in rows):
        low_count = sum(
            1
            for r in rows
            if r.get("keyword_hit_rate") is not None
            and r.get("keyword_hit_rate") < keyword_threshold
        )
        if low_count:
            insights.append(f"{low_count}/{total} have low keyword hit.")
            low_fail = sum(
                1
                for r in rows
                if r.get("keyword_hit_rate") is not None
                and r.get("keyword_hit_rate") < keyword_threshold
                and _is_fail(r)
            )
            high_count = sum(
                1
                for r in rows
                if r.get("keyword_hit_rate") is not None
                and r.get("keyword_hit_rate") >= keyword_threshold
            )
            if high_count > 0 and low_fail / max(1, low_count) > 0.0:
                insights.append("Low keyword hit aligns with more failures.")

    if mode == "formal" and any(r.get("refusal_correct") is not None for r in rows):
        refusal_errors = sum(1 for r in rows if r.get("refusal_correct") is False)
        if refusal_errors:
            insights.append(f"{refusal_errors} refusal errors detected.")
            should_refuse_errors = sum(
                1
                for r in rows
                if r.get("refusal_correct") is False and r.get("category") == "should_refuse"
            )
            if should_refuse_errors and should_refuse_errors / max(1, refusal_errors) >= 0.5:
                insights.append("Refusal errors are concentrated in should_refuse items.")

    if missing_citations:
        recommendations.append(
            "Improve citation consistency or require at least 1 cited source."
        )
    if low_keyword:
        recommendations.append(
            "Refine expected keywords or add synonyms for better relevance checks."
        )
    if slow_count:
        recommendations.append(
            "Reduce TOP_K or disable judge metrics for faster routine runs."
        )
    if refusal_incorrect and mode == "formal":
        recommendations.append(
            "Add more should_refuse items and tighten refusal guidance."
        )

    if not recommendations:
        recommendations.append("Results look stable; consider expanding the evalset.")

    insights = [i for i in insights if len(i) <= 120]
    recommendations = [r for r in recommendations if len(r) <= 120]

    return insights, recommendations


st.title("3) Evaluate")
st.write(
    "Run a formal evaluation set against the same RAG pipeline used in Chat, "
    "or quickly diagnose recent chat questions."
)

ensure_prefs()
apply_accent_css(get_pref("accent"))

mode = st.radio("Mode", options=[MODE_FORMAL, MODE_CHAT], horizontal=True)
chat_mode = mode == MODE_CHAT

default_path = CFG.EVALSETS_DIR / "evalset.json"

st.subheader("Configure")

if chat_mode:
    chat_n = st.number_input(
        "Number of recent chat questions",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )
    timeout_s = st.number_input(
        "Per-question timeout (seconds)",
        min_value=5,
        max_value=300,
        value=60,
        step=5,
    )
    path_str = st.text_input(
        "Evalset file path (used for Promote to evalset)",
        value=str(default_path),
    )
else:
    with st.expander("Evalset format (example)"):
        st.code(
            """
[
  {
    "id": "A1",
    "category": "answerable",
    "question": "What is the return policy?",
    "expected_keywords": ["return", "refund", "receipt"],
    "min_citations": 1,
    "should_refuse": false
  },
  {
    "id": "B1",
    "category": "should_refuse",
    "question": "What is the capital of France?",
    "should_refuse": true
  }
]
            """.strip(),
            language="json",
        )

    path_str = st.text_input("Evalset file path", value=str(default_path))
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        enable_judge = st.checkbox("Enable LLM judge metrics (costs tokens)", value=False)
    with col2:
        timeout_s = st.number_input(
            "Per-question timeout (seconds)",
            min_value=5,
            max_value=300,
            value=60,
            step=5,
        )

st.subheader("Run")

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "chat_eval_results" not in st.session_state:
    st.session_state.chat_eval_results = None

run_clicked = st.button("Run evaluation")

if run_clicked:
    if chat_mode:
        scope_key, scope_label, messages = _get_active_chat_messages()
        if not scope_key or messages is None:
            st.error("No active chat history found. Ask a question in Chat first.")
            st.stop()

        user_questions = [m["content"] for m in messages if m.get("role") == "user"]
        user_questions = user_questions[-int(chat_n) :]

        if not user_questions:
            st.error("No user questions found in the active chat history.")
            st.stop()

        if scope_label and scope_label != "All PDFs":
            allowed_sources = [scope_label]
        else:
            allowed_sources = None

        with st.spinner("Running chat-history evaluation..."):
            results = run_chat_history_evaluation(
                user_questions,
                allowed_sources=allowed_sources,
                timeout_s=int(timeout_s),
            )
        st.session_state.chat_eval_results = {
            "scope_label": scope_label,
            "summary": results["summary"],
            "items": results["items"],
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
    else:
        path = Path(path_str)
        try:
            items = load_evalset(path)
        except FileNotFoundError:
            st.error("Evalset file not found. Create data/evalsets/evalset.json first.")
            st.stop()
        except ValueError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Running evaluation..."):
            results = run_evaluation(
                items,
                judge_enabled=enable_judge,
                timeout_s=int(timeout_s),
                evalset_path=path,
            )
        st.session_state.eval_results = results

with st.expander("What this evaluation tests", expanded=False):
    st.markdown("**Grounding & Evidence**")
    st.markdown("- Checks that answers point to supporting sources.")
    st.markdown("- Flags when citations are missing or weak.")

    st.markdown("**Relevance & Correctness signals**")
    st.markdown("- Keyword coverage (formal mode) indicates topic match.")
    st.markdown("- Refusal behavior should align with the question.")

    st.markdown("**Performance**")
    st.markdown("- Measures response latency for each question.")
    st.markdown("- Highlights slow cases with P95 latency.")

with st.expander("Metrics glossary", expanded=False):
    for label, info in METRIC_GLOSSARY.items():
        definition = info.get("definition")
        if definition:
            st.markdown(f"**{label}**: {definition}")

st.subheader("Results")

if chat_mode:
    results = st.session_state.chat_eval_results
    if results:
        summary = results["summary"]
        st.caption(f"Scope: {results.get('scope_label') or 'All PDFs'}")

        c1, c2, c3, c4, c5 = st.columns(5)
        _metric_card(c1, "Citation coverage", summary.get("citation_coverage_rate", 0.0))
        _metric_card(c2, "Refusal rate", summary.get("refusal_rate", 0.0))
        _metric_card(c3, "Refusal count", summary.get("refusal_count", 0))
        _metric_card(c4, "Median latency (s)", summary.get("median_latency", 0.0))
        _metric_card(c5, "P95 latency (s)", summary.get("p95_latency", 0.0))

        items = results.get("items", [])
        highlight_slow = bool(get_pref("eval_highlight_slow"))
        slow_threshold = float(get_pref("eval_slow_threshold_s"))
        keyword_threshold = float(get_pref("eval_keyword_threshold"))

        base_rows = []
        for item in items:
            status = _status_chat(item)
            slow_flag = ""
            if highlight_slow and item.get("latency_seconds", 0.0) >= slow_threshold:
                slow_flag = "⚠ slow"
            base_rows.append(
                {
                    "question": item.get("question"),
                    "status": status,
                    "refused": item.get("refused"),
                    "latency_seconds": item.get("latency_seconds"),
                    "citation_ok": item.get("citation_ok"),
                    "slow_flag": slow_flag,
                    "error": item.get("error"),
                }
            )

        insights, recommendations = _build_insights(
            base_rows,
            slow_threshold=slow_threshold,
            keyword_threshold=keyword_threshold,
            mode="chat",
        )
        with st.expander("Evaluation insights", expanded=False):
            st.caption(
                "Automatic interpretation of this run to help identify weak areas and next steps."
            )
            st.markdown("**Key findings**")
            for i in insights:
                st.markdown(f"- {i}")
            st.markdown("**Recommended next steps**")
            for r in recommendations:
                st.markdown(f"- {r}")

        st.subheader("Per-Question Results")
        failed_default = bool(get_pref("eval_show_failed_only"))
        show_failed_only = st.checkbox("Show failed only", value=failed_default)
        sort_choice = st.selectbox(
            "Sort by",
            ["Latency (desc)", "Latency (asc)", "Citation ok (fail first)", "Question (A-Z)"],
        )

        display_rows = base_rows[:]
        if show_failed_only:
            display_rows = [
                i
                for i in display_rows
                if (i.get("citation_ok") is False) or i.get("error")
            ]
        if sort_choice == "Latency (asc)":
            display_rows = sorted(display_rows, key=lambda x: x.get("latency_seconds", 0.0))
        elif sort_choice == "Latency (desc)":
            display_rows = sorted(display_rows, key=lambda x: x.get("latency_seconds", 0.0), reverse=True)
        elif sort_choice == "Citation ok (fail first)":
            display_rows = sorted(display_rows, key=lambda x: x.get("citation_ok") is True)
        elif sort_choice == "Question (A-Z)":
            display_rows = sorted(display_rows, key=lambda x: (x.get("question") or "").lower())

        st.dataframe(display_rows, use_container_width=True)

        st.subheader("Download Results")
        _ts = results.get("timestamp", time.strftime("%Y%m%d_%H%M%S"))
        _dl_rows = []
        for item in results.get("items", []):
            _dl_rows.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "refused": item.get("refused"),
                "latency_seconds": item.get("latency_seconds"),
                "citation_ok": item.get("citation_ok"),
                "slow_flag": (item.get("latency_seconds") or 0.0) >= slow_threshold,
                "timestamp": _ts,
            })
        _json_bytes = json.dumps(_dl_rows, indent=2).encode("utf-8")
        st.download_button(
            "Download JSON",
            data=_json_bytes,
            file_name=f"chat_diagnostics_{_ts}.json",
            mime="application/json",
        )
        _csv_buf = io.StringIO()
        if _dl_rows:
            _writer = csv.DictWriter(_csv_buf, fieldnames=list(_dl_rows[0].keys()))
            _writer.writeheader()
            _writer.writerows(_dl_rows)
        st.download_button(
            "Download CSV",
            data=_csv_buf.getvalue().encode("utf-8"),
            file_name=f"chat_diagnostics_{_ts}.csv",
            mime="text/csv",
        )

        st.subheader("Promote to Evalset")
        st.caption("Add any of these questions to your labelled evalset.json.")
        evalset_path = Path(path_str)

        for idx, item in enumerate(results.get("items", [])):
            cols = st.columns([0.45, 0.15, 0.2, 0.2])
            cols[0].write(item.get("question"))
            category = cols[1].selectbox(
                "Category",
                options=["answerable", "should_refuse", "confusable"],
                key=f"cat_{idx}",
            )
            keywords_raw = cols[2].text_input(
                "Expected keywords (comma)",
                key=f"kw_{idx}",
            )
            promote = cols[3].button("Promote to evalset", key=f"promote_{idx}")

            if promote:
                expected_keywords = [
                    k.strip()
                    for k in (keywords_raw or "").split(",")
                    if k.strip()
                ]
                item_id = f"CHAT_{int(time.time())}_{idx}"
                new_item = {
                    "id": item_id,
                    "category": category,
                    "question": item.get("question"),
                }
                if category == "should_refuse":
                    new_item["should_refuse"] = True
                else:
                    new_item["should_refuse"] = False
                    new_item["min_citations"] = 1
                    if expected_keywords:
                        new_item["expected_keywords"] = expected_keywords

                _append_to_evalset(new_item, evalset_path)
                st.success(f"Added {item_id} to {evalset_path}")
else:
    results = st.session_state.eval_results
    if results:
        summary = results["summary"]
        judge_used = results.get("meta", {}).get("judge_enabled", False)

        c1, c2, c3, c4, c5 = st.columns(5)
        _metric_card(c1, "Refusal accuracy", summary.get("refusal_accuracy", 0.0))
        _metric_card(c2, "Citation coverage", summary.get("citation_coverage_rate", 0.0))
        _metric_card(c3, "Avg keyword hit rate", summary.get("avg_keyword_hit_rate", 0.0))
        _metric_card(c4, "Median latency (s)", summary.get("median_latency", 0.0))
        _metric_card(c5, "P95 latency (s)", summary.get("p95_latency", 0.0))

        if judge_used:
            st.caption("LLM judge metrics (see glossary)")
            j1, j2, j3 = st.columns(3)
            _metric_card(j1, "Avg groundedness", summary.get("avg_groundedness", 0.0))
            _metric_card(j2, "Avg context relevance", summary.get("avg_context_relevance", 0.0))
            _metric_card(j3, "Avg answer relevance", summary.get("avg_answer_relevance", 0.0))

        items_all = results.get("items", [])
        evalset_lookup = _load_evalset_lookup(Path(path_str))
        highlight_slow = bool(get_pref("eval_highlight_slow"))
        slow_threshold = float(get_pref("eval_slow_threshold_s"))
        highlight_low_keyword = bool(get_pref("eval_highlight_low_keyword"))
        keyword_threshold = float(get_pref("eval_keyword_threshold"))

        base_items = []
        for i in items_all:
            expected = evalset_lookup.get(i.get("id")) or []
            i["_status"] = _status_formal(i, expected_keywords_present=len(expected) > 0)
            i["_slow_flag"] = ""
            i["_low_keyword_flag"] = ""
            if highlight_slow and i.get("latency_seconds", 0.0) >= slow_threshold:
                i["_slow_flag"] = "⚠ slow"
            if highlight_low_keyword:
                rate = (i.get("metrics") or {}).get("keyword_hit_rate")
                if rate is not None and rate < keyword_threshold:
                    i["_low_keyword_flag"] = "⚠ low"
            base_items.append(i)

        base_rows = []
        for item in base_items:
            base_rows.append(
                {
                    "id": item.get("id"),
                    "category": item.get("category"),
                    "question": item.get("question"),
                    "status": item.get("_status"),
                    "refused": item.get("refused"),
                    "latency_seconds": item.get("latency_seconds"),
                    "keyword_hit_rate": item.get("metrics", {}).get("keyword_hit_rate"),
                    "citation_ok": item.get("metrics", {}).get("citation_ok"),
                    "refusal_correct": item.get("metrics", {}).get("refusal_correct"),
                    "slow_flag": item.get("_slow_flag"),
                    "low_keyword": item.get("_low_keyword_flag"),
                    "error": item.get("error"),
                }
            )

        insights, recommendations = _build_insights(
            base_rows,
            slow_threshold=slow_threshold,
            keyword_threshold=keyword_threshold,
            mode="formal",
        )
        with st.expander("Evaluation insights", expanded=False):
            st.caption(
                "Automatic interpretation of this run to help identify weak areas and next steps."
            )
            st.markdown("**Key findings**")
            for i in insights:
                st.markdown(f"- {i}")
            st.markdown("**Recommended next steps**")
            for r in recommendations:
                st.markdown(f"- {r}")

        st.subheader("Breakdown by Category")
        st.dataframe(results.get("by_category", []), use_container_width=True)

        st.subheader("Per-Question Results")
        items = base_items[:]
        categories = sorted({i.get("category") for i in items if i.get("category")})
        col_a, col_b, col_c = st.columns([0.35, 0.35, 0.3])
        failed_default = bool(get_pref("eval_show_failed_only"))
        failed_only = col_a.checkbox("Show failed only", value=failed_default)
        category_filter = col_b.selectbox("Category filter", options=["All"] + categories)
        sort_choice = col_c.selectbox(
            "Sort by",
            [
                "Latency (desc)",
                "Latency (asc)",
                "Keyword hit rate (desc)",
                "Status",
                "ID",
            ],
        )

        if failed_only:
            items = [i for i in items if i.get("failed")]

        if category_filter != "All":
            items = [i for i in items if i.get("category") == category_filter]

        if sort_choice == "Latency (asc)":
            items = sorted(items, key=lambda x: x.get("latency_seconds", 0.0))
        elif sort_choice == "Latency (desc)":
            items = sorted(items, key=lambda x: x.get("latency_seconds", 0.0), reverse=True)
        elif sort_choice == "Keyword hit rate (desc)":
            items = sorted(
                items,
                key=lambda x: (x.get("metrics") or {}).get("keyword_hit_rate", 0.0),
                reverse=True,
            )
        elif sort_choice == "Status":
            items = sorted(items, key=lambda x: x.get("_status", "pass"))
        elif sort_choice == "ID":
            items = sorted(items, key=lambda x: x.get("id") or "")

        table_rows = []
        for item in items:
            row = {
                "id": item.get("id"),
                "category": item.get("category"),
                "question": item.get("question"),
                "status": item.get("_status"),
                "refused": item.get("refused"),
                "latency_seconds": item.get("latency_seconds"),
                "keyword_hit_rate": item.get("metrics", {}).get("keyword_hit_rate"),
                "citation_ok": item.get("metrics", {}).get("citation_ok"),
                "refusal_correct": item.get("metrics", {}).get("refusal_correct"),
                "slow_flag": item.get("_slow_flag"),
                "low_keyword": item.get("_low_keyword_flag"),
                "error": item.get("error"),
            }
            table_rows.append(row)

        st.dataframe(table_rows, use_container_width=True)

        st.subheader("Download Results")
        artifacts = results.get("artifacts", {})
        json_path = artifacts.get("json_path")
        csv_path = artifacts.get("csv_path")
        if json_path and Path(json_path).exists():
            st.download_button(
                "Download JSON",
                data=Path(json_path).read_bytes(),
                file_name=Path(json_path).name,
                mime="application/json",
            )
        if csv_path and Path(csv_path).exists():
            st.download_button(
                "Download CSV",
                data=Path(csv_path).read_bytes(),
                file_name=Path(csv_path).name,
                mime="text/csv",
            )

        if json_path or csv_path:
            st.caption(
                f"Output paths: {json_path or 'n/a'}, {csv_path or 'n/a'}"
            )
