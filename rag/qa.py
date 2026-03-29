from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from .config import CFG, assert_api_key
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .store import similarity_search_with_score


def _llm() -> ChatOpenAI:
    assert_api_key()
    return ChatOpenAI(model=CFG.CHAT_MODEL, temperature=0)


def _format_context(pairs: List[Tuple[Document, float]]) -> str:
    blocks = []
    for d, dist in pairs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[Source: {src}, p.{page}]\n{(d.page_content or '').strip()}")
    return "\n\n---\n\n".join(blocks)


def _sources(pairs: List[Tuple[Document, float]]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for d, dist in pairs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)

        snippet = (d.page_content or "").strip().replace("\n", " ")
        snippet = snippet[:200] + ("..." if len(snippet) > 200 else "")
        chunk = d.metadata.get("chunk", None)
        out.append(
            {
                "source": src,
                "page": page,
                "chunk": chunk,
                "distance": float(dist) if dist is not None else None,
                "snippet": snippet,
            }
        )
    return out


def _question_keywords(question: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9']+", question.lower())
    return [w for w in words if len(w) >= 4]


def _classify_intent(question: str) -> str:
    q = (question or "").lower().strip()
    _factual = re.search(
        r"\b(point|points|rebound|rebounds|assist|assists|average|averages|"
        r"scored|ppg|apg|rpg|win|wins|loss|losses|game|games|season|"
        r"goal|goals|score|scores|year|years|calorie|calories|"
        r"mile|miles|percent|percentage|degree|degrees)\b",
        q,
    )
    if _factual:
        return "specific"
    # Only treat as "count" when asking about items/entries IN a list or document,
    # not for factual "how many" questions (scores, stats, credits, averages, etc.).
    _count_trigger = re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b", q)
    _list_context = re.search(
        r"\bitems?\b|\bproducts?\b|\bentries\b|\bentry\b|\blisted\b|\bon the menu\b"
        r"|\boptions?\b|\bchoices?\b|\bdishes?\b|\bthings?\b on|\bin the (menu|list|document|pdf)\b",
        q,
    )
    if _count_trigger and _list_context:
        return "count"
    broad_markers = [
        "what is in the pdf",
        "what's in the pdf",
        "what is in this pdf",
        "what's in this pdf",
        "what is in the document",
        "what's in the document",
        "summarise",
        "summarize",
        "overview",
        "describe the document",
        "tell me about the document",
        "tell me about this",
        "what is this document",
        "list everything",
        "list all",
    ]
    if any(p in q for p in broad_markers):
        return "broad"
    # Heuristic: very short, generic questions about the document itself
    words = re.findall(r"[a-zA-Z0-9']+", q)
    if len(words) <= 6 and any(w in {"pdf", "document", "file", "menu"} for w in words):
        return "broad"
    return "specific"


def _extract_headings_and_sections(texts: List[str]) -> List[str]:
    headings: List[str] = []
    seen = set()
    for text in texts:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if len(line) > 60:
                continue
            if line.endswith(":"):
                pass_check = True
            elif line.isupper() and len(line) <= 60:
                pass_check = True
            elif line == line.title() and len(line.split()) <= 8:
                pass_check = True
            else:
                pass_check = False
            if not pass_check:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            headings.append(line)
            if len(headings) >= 12:
                return headings
    return headings


def _build_overview_answer(headings: List[str]) -> str:
    if headings:
        bullets = "\n".join([f"- {h}" for h in headings])
        return (
            "I can see the document/menu contains these sections:\n"
            f"{bullets}\n\n"
            "Tip: Ask a narrower question like 'List all sandwiches' or "
            "'What is in Hot Picks?'"
        )
    return (
        "I found relevant text, but it doesn't clearly list sections. "
        "Try asking about a specific category or item."
    )


def _looks_like_menu_item(line: str) -> bool:
    text = (line or "").strip()
    if not text:
        return False
    if text.endswith(":"):
        return False
    if text.isupper() and len(text) <= 60:
        return False
    if re.search(r"£\s?\d+(\.\d{2})?|\b\d+\.\d{2}\b", text):
        return True
    if re.match(r"^\s*[\-\u2022•\*]\s+", text):
        return True
    if re.match(r"^\s*\d+[\).\s]+", text):
        return True
    if " - " in text:
        return True
    if "," in text and len(text) <= 120:
        return True
    return False


def _count_items_in_texts(texts: List[str]) -> Tuple[int, List[str]]:
    seen = set()
    examples: List[str] = []
    for text in texts:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not _looks_like_menu_item(line):
                continue
            norm = re.sub(r"\s+", " ", line).strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            if len(examples) < 5:
                examples.append(line)
    return len(seen), examples


def _with_refusal_tip(message: str) -> str:
    tip = (
        "Tip: Try asking about a specific section, item name, or include a keyword from the document."
    )
    if tip in message:
        return message
    return f"{message}\n{tip}"


def _is_price_extreme_question(question: str) -> Optional[str]:
    q = (question or "").lower()
    if re.search(r"\bmost expensive\b|\bhighest price\b|\bpriciest\b|\bmost costly\b", q):
        return "max"
    if re.search(r"\bcheapest\b|\blowest price\b|\bleast expensive\b|\blowest cost\b", q):
        return "min"
    return None


def _is_category_header(line: str) -> bool:
    text = (line or "").strip()
    if not text:
        return False
    if text.endswith(":"):
        return True
    if text.isupper() and len(text) <= 60:
        return True
    return False


def _detect_category_constraint(question: str) -> Optional[str]:
    q = (question or "").lower()
    if "signature" in q and "juice" in q:
        return "signature_juices"
    if "protein" in q and "shake" in q:
        return "protein_shakes"
    if "juice" in q or "juices" in q:
        return "juices"
    if "shake" in q or "shakes" in q:
        return "shakes"
    if "sandwich" in q or "sandwiches" in q:
        return "sandwiches"
    if "salad" in q or "salad bowls" in q or "bowl" in q:
        return "salad_bowls"
    if "hot pick" in q or "hot picks" in q:
        return "hot_picks"
    if "breakfast" in q:
        return "breakfast"
    if "shot" in q or "shots" in q:
        return "shots"
    if "water" in q:
        return "water"
    return None


def _normalize_section_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return key


def _sectionize(texts: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    for text in texts:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            is_heading = False
            if len(line) <= 60:
                if line.endswith(":"):
                    is_heading = True
                elif line.isupper():
                    is_heading = True
                elif line == line.title() and len(line.split()) <= 8:
                    is_heading = True
            if is_heading:
                current_key = _normalize_section_key(line.rstrip(":"))
                sections.setdefault(current_key, [])
                continue
            if current_key is None:
                current_key = "unknown"
                sections.setdefault(current_key, [])
            sections[current_key].append(line)
    return sections


def _category_keyword(category: str) -> str:
    mapping = {
        "signature_juices": "juice",
        "juices": "juice",
        "protein_shakes": "protein",
        "shakes": "shake",
        "sandwiches": "sandwich",
        "salad_bowls": "salad",
        "hot_picks": "hot",
        "breakfast": "breakfast",
        "shots": "shot",
        "water": "water",
    }
    return mapping.get(category, category.replace("_", " "))


def _find_extreme_priced_items_in_category(
    texts: List[str], category: str, mode: str
) -> List[Tuple[str, float, str]]:
    sections = _sectionize(texts)
    key = _normalize_section_key(category)
    lines = sections.get(key)
    results: List[Tuple[str, float, str]] = []

    def _extract_from_lines(lines_in: List[str]) -> List[Tuple[str, float, str]]:
        items: List[Tuple[str, float, str]] = []
        price_re = re.compile(r"£\s?(\d+(?:\.\d{1,2})?)")
        last_name: Optional[str] = None
        for raw in lines_in:
            line = raw.strip()
            if not line:
                continue
            if _is_category_header(line):
                continue
            matches = list(price_re.finditer(line))
            if not matches:
                last_name = line
                continue
            same_line_name = price_re.sub("", line)
            same_line_name = re.sub(r"\s{2,}", " ", same_line_name).strip(" -–—|•·")
            if same_line_name and re.search(r"[A-Za-z]", same_line_name):
                candidate_name = same_line_name
            else:
                candidate_name = last_name
            if not candidate_name:
                continue
            ingredient_line = line
            if "," not in ingredient_line and last_name and "," in last_name:
                ingredient_line = last_name
            for m in matches:
                try:
                    price = float(m.group(1))
                except ValueError:
                    continue
                items.append((candidate_name, price, ingredient_line))
        return items

    if lines:
        results = _extract_from_lines(lines)
    else:
        keyword = _category_keyword(category)
        prev_lines: List[str] = []
        scoped: List[str] = []
        for text in texts:
            for raw in (text or "").splitlines():
                line = raw.strip()
                if not line:
                    continue
                window = " ".join(prev_lines + [line]).lower()
                if keyword in window:
                    scoped.append(line)
                prev_lines = (prev_lines + [line])[-2:]
        if scoped:
            results = _extract_from_lines(scoped)

    if not results:
        return []

    prices = [p for _, p, _ in results]
    target = max(prices) if mode == "max" else min(prices)
    return [(n, p, l) for n, p, l in results if p == target]


def _extract_priced_items(texts: List[str]) -> List[Tuple[str, float]]:
    priced: List[Tuple[str, float]] = []
    price_re = re.compile(r"£\s?(\d+(?:\.\d{1,2})?)")
    last_name: Optional[str] = None

    for text in texts:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if _is_category_header(line):
                continue

            matches = list(price_re.finditer(line))
            if not matches:
                last_name = line
                continue

            # Try to use the name from the same line if it includes words.
            same_line_name = price_re.sub("", line)
            same_line_name = re.sub(r"\s{2,}", " ", same_line_name).strip(" -–—|•·")
            if same_line_name and re.search(r"[A-Za-z]", same_line_name):
                candidate_name = same_line_name
            else:
                candidate_name = last_name

            if not candidate_name:
                continue

            for m in matches:
                try:
                    price = float(m.group(1))
                except ValueError:
                    continue
                priced.append((candidate_name, price))

    return priced


def _count_ingredients_from_item_line(item_line: str) -> Optional[int]:
    if not item_line:
        return None
    cleaned = re.sub(r"£\s?\d+(?:\.\d{1,2})?", "", item_line)
    if "," not in cleaned:
        return None
    parts = [p.strip(" -–—|•·") for p in cleaned.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return len(parts)


def _build_debug(
    sources: List[Dict],
    contexts: List[str],
    refused: bool,
    retrieval_debug: Optional[List[Dict]] = None,
) -> Dict:
    if retrieval_debug is None:
        retrieval_debug = [
            {
                "doc": s.get("source"),
                "page": s.get("page"),
                "chunk_id": s.get("chunk"),
                "distance": s.get("distance"),
            }
            for s in sources
        ]
    return {
        "refused": refused,
        "retrieval_debug": retrieval_debug,
        "retrieved_contexts": contexts,
    }


def answer_question(
    question: str,
    k: Optional[int] = None,
    allowed_sources: Optional[List[str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    style_hint: Optional[str] = None,
    return_debug: bool = False,
) -> Union[Tuple[str, List[Dict]], Tuple[str, List[Dict], Dict]]:
    intent = _classify_intent(question)
    effective_k = k or CFG.TOP_K
    if intent == "broad":
        effective_k = max(effective_k, CFG.TOP_K * 4)
    elif intent == "count":
        effective_k = max(effective_k, CFG.TOP_K * 6)
    else:
        effective_k = max(effective_k, CFG.TOP_K * 2)

    pairs = similarity_search_with_score(
        query=question,
        k=effective_k,
        sources=allowed_sources,
    )

    # If retrieval returns nothing, it usually means:
    # - vectorstore empty OR
    # - the uploaded doc had no text (needs OCR) OR
    # - wrong collection/path
    def _final(
        answer_text: str,
        sources: List[Dict],
        contexts: List[str],
        refused: bool,
        retrieval_debug: Optional[List[Dict]] = None,
    ):
        debug = _build_debug(sources, contexts, refused, retrieval_debug)
        if return_debug:
            return answer_text, sources, debug
        return answer_text, sources

    if not pairs:
        return _final(
            _with_refusal_tip(
                "I couldn't find anything relevant in the uploaded files. "
                "If the PDF is scanned/image-based, it may not be searchable without OCR. "
                "Try uploading a text-based PDF."
            ),
            [],
            [],
            True,
        )

    # Clean + sort by distance (lower is better)
    clean = [(d, float(dist)) for (d, dist) in pairs if dist is not None]
    if not clean:
        sources = _sources(pairs)
        return _final(
            _with_refusal_tip(
                "I couldn't find anything reliable in the uploaded files for this question."
            ),
            sources,
            [],
            True,
        )

    clean.sort(key=lambda x: x[1])

    # Optional distance filter, but never hard-fail if it filters everything.
    max_dist = getattr(CFG, "MAX_DISTANCE", 0)
    if max_dist and max_dist > 0:
        good = [(d, dist) for (d, dist) in clean if dist <= max_dist]
    else:
        good = clean

    if not good:
        # Fall back to the closest chunks and let the LLM decide based on context.
        good = clean[: (k or CFG.TOP_K)]

    # Simple confidence checks (skip for broad summary questions)
    contexts = [(d.page_content or "").strip() for d, _ in good]
    retrieval_debug = [
        {
            "doc": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "chunk_id": d.metadata.get("chunk"),
            "distance": dist,
        }
        for d, dist in good
    ]
    context_text = " ".join(contexts).strip()
    best_dist = clean[0][1] if clean else None
    context_is_short = len(context_text) < CFG.MIN_CONTEXT_CHARS and (
        best_dist is None or best_dist > max_dist
    )
    keywords = _question_keywords(question)
    keyword_hits = None
    if len(keywords) >= CFG.MIN_KEYWORDS_FOR_CHECK:
        context_lower = context_text.lower()
        keyword_hits = sum(1 for w in keywords if w in context_lower)

    extreme = _is_price_extreme_question(question)
    ingredient_count_query = bool(
        re.search(
            r"\bhow many ingredients\b|\bingredient count\b|\bnumber of ingredients\b|"
            r"\bhow many items in\b",
            (question or "").lower(),
        )
    )
    if extreme in {"max", "min"} and ingredient_count_query:
        category = _detect_category_constraint(question)
        sources = _sources(good)
        if not category:
            return _final(
                _with_refusal_tip(
                    "I couldn't tell which category you meant (e.g., juices, sandwiches). "
                    "Please specify a category."
                ),
                sources,
                contexts,
                True,
                retrieval_debug,
            )
        items = _find_extreme_priced_items_in_category(contexts, category, extreme)
        if not items:
            return _final(
                _with_refusal_tip(
                    "I couldn't find priced items for that category in the retrieved text. "
                    "Try asking about a different category or re-upload the document."
                ),
                sources,
                contexts,
                True,
                retrieval_debug,
            )
        seen = set()
        unique: List[Tuple[str, float, str]] = []
        for name, price, line in items:
            key = (name.lower(), price)
            if key in seen:
                continue
            seen.add(key)
            unique.append((name, price, line))
        label = "most expensive" if extreme == "max" else "cheapest"
        category_label = category.replace("_", " ")
        bullets = []
        for name, price, line in unique:
            count = _count_ingredients_from_item_line(line)
            if count is None:
                count_text = "ingredients not clearly listed"
            else:
                count_text = f"{count} ingredients"
            bullets.append(f"- {name} (£{price:.2f}) — {count_text}")
        answer = (
            f"For the {label} {category_label}, I found:\n"
            + "\n".join(bullets)
            + "\n\n"
            "Note: This is based on the retrieved sections and may be incomplete if the PDF is partial."
        )
        return _final(answer, sources, contexts, False, retrieval_debug)

    if extreme in {"max", "min"}:
        category = _detect_category_constraint(question)
        sources = _sources(good)
        if category:
            items_in_category = _find_extreme_priced_items_in_category(
                contexts, category, extreme
            )
            if not items_in_category:
                return _final(
                    _with_refusal_tip(
                        "I couldn't find priced items for that category in the retrieved text. "
                        "Try asking about a different category or re-upload the document."
                    ),
                    sources,
                    contexts,
                    True,
                    retrieval_debug,
                )
            items = [(n, p) for n, p, _ in items_in_category]
        else:
            items = _extract_priced_items(contexts)
        if items:
            seen = set()
            unique: List[Tuple[str, float]] = []
            for name, price in items:
                key = (name.lower(), price)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((name, price))
            prices = [p for _, p in unique]
            target = max(prices) if extreme == "max" else min(prices)
            tied = [(n, p) for n, p in unique if p == target]
            label = "most expensive" if extreme == "max" else "cheapest"
            bullets = "\n".join([f"- {n} (£{p:.2f})" for n, p in tied])
            if category:
                answer = (
                    f"The {label} {category.replace('_', ' ')} I can see in the retrieved text are:\n"
                    f"{bullets}\n\n"
                    "Note: This is based on the retrieved sections and may be incomplete if the PDF is partial."
                )
            else:
                answer = (
                    f"The {label} item(s) I can see in the retrieved text are:\n"
                    f"{bullets}\n\n"
                    "Note: This is based on the retrieved sections and may be incomplete if the PDF is partial."
                )
            return _final(answer, sources, contexts, False, retrieval_debug)
        if category:
            return _final(
                _with_refusal_tip(
                    "I couldn't find priced items for that category in the retrieved text. "
                    "Try asking about a different category or re-upload the document."
                ),
                sources,
                contexts,
                True,
                retrieval_debug,
            )

    if intent == "broad":
        if context_is_short or (keyword_hits == 0):
            headings = _extract_headings_and_sections(contexts)
            overview = _build_overview_answer(headings)
            sources = _sources(good)
            return _final(overview, sources, contexts, False, retrieval_debug)
    elif intent == "specific":
        if context_is_short:
            sources = _sources(good)
            return _final(
                _with_refusal_tip(
                    "I couldn't find enough relevant content in the uploads to answer confidently. "
                    "Try rephrasing the question or upload a more detailed document."
                ),
                sources,
                contexts,
                True,
                retrieval_debug,
            )
        if keyword_hits == 0:
            sources = _sources(good)
            return _final(
                _with_refusal_tip(
                    "No relevant content found in your uploads for that question. "
                    "Try rephrasing or upload a more specific document."
                ),
                sources,
                contexts,
                True,
                retrieval_debug,
            )

    if intent == "count":
        count, _examples = _count_items_in_texts(contexts)
        if count > 0:
            answer = f"I counted {count} item lines in the retrieved menu text."
        else:
            answer = "I couldn't reliably identify item lines to count from the retrieved text."
        answer = (
            f"{answer}\n\n"
            "Note: This count is based on the retrieved sections and may be incomplete if the PDF is partial.\n"
            "Tip: For a precise count, ask per category (e.g., 'How many Sandwiches are listed?')"
        )
        sources = _sources(good)
        return _final(answer, sources, contexts, False, retrieval_debug)

    context = _format_context(good)
    sources = _sources(good)

    system_prompt = SYSTEM_PROMPT
    if style_hint:
        system_prompt = f"{SYSTEM_PROMPT}\n\nStyle guide: {style_hint}"

    messages = [("system", system_prompt)]
    # history accepted for future multi-turn support; not yet passed to LLM
    if history:
        # Keep only well-formed roles
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append((role, content))

    messages.append(
        ("user", USER_PROMPT_TEMPLATE.format(question=question, context=context))
    )

    resp = _llm().invoke(messages)
    return _final(resp.content, sources, contexts, False, retrieval_debug)
