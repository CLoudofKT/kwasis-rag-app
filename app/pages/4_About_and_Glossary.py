import streamlit as st

from app.ui_prefs import apply_accent_css, ensure_prefs, get_pref

st.title("About the System & Evaluation Framework")
st.caption(
    "This page explains how the application works, how answers are evaluated, "
    "and how users can interpret and improve results."
)

ensure_prefs()
apply_accent_css(get_pref("accent"))

with st.expander("What does this application do?", expanded=False):
    st.markdown("- Allows users to upload one or more PDF documents.")
    st.markdown("- Answers questions only using the uploaded documents.")
    st.markdown("- Displays sources alongside each answer for transparency.")
    st.markdown("- Refuses to answer when information is not present in the documents.")
    st.caption("The system is designed to prioritise accuracy and transparency over guesswork.")

with st.expander("How are answers generated?", expanded=False):
    st.markdown("1. Uploaded documents are split into smaller sections.")
    st.markdown("2. Relevant sections are retrieved for each question.")
    st.markdown("3. The answer is generated using only the retrieved content.")
    st.markdown("4. Sources are attached to each response for verification.")
    st.caption("This approach is known as Retrieval-Augmented Generation (RAG).")

with st.expander("How does evaluation work?", expanded=False):
    st.write(
        "The system evaluates answers across three dimensions: accuracy, relevance, and performance."
    )

    st.markdown("**Accuracy**")
    st.markdown("- Checks whether the system answers when the information exists in the PDFs.")
    st.markdown("- Checks whether the system refuses when the information is not present.")
    st.markdown("- Helps identify hallucinations and unsafe answers.")

    st.markdown("**Relevance**")
    st.markdown("- Measures whether answers contain expected concepts or terms.")
    st.markdown("- Acts as a lightweight signal for topical alignment.")
    st.markdown("- Does not reliably capture paraphrasing or synonyms.")

    st.markdown("**Performance**")
    st.markdown("- Measures how long the system takes to respond.")
    st.markdown("- Includes typical speed and worst-case latency.")
    st.markdown("- Helps assess usability as document size and query volume increases.")

with st.expander("Metrics glossary", expanded=False):
    st.markdown("### Accuracy metrics")
    st.markdown(
        "- Refusal accuracy: Proportion of cases where the system refused only when it should have."
    )
    st.markdown(
        "- Refusal rate: Share of questions where the system declined to answer."
    )
    st.markdown("- Refusal count: Total number of refusals in a single evaluation run.")
    st.markdown(
        "- Refusal correct: Whether an individual refusal was appropriate for that question."
    )

    st.markdown("### Relevance metrics")
    st.markdown(
        "- Citation coverage: Proportion of answers that include at least one cited source."
    )
    st.markdown("- Citation OK: Whether an answer met the minimum citation requirement.")
    st.markdown(
        "- Keyword hit rate: Fraction of expected keywords found in a single answer."
    )
    st.markdown(
        "- Avg keyword hit rate: Average keyword hit rate across all evaluated questions."
    )
    st.markdown(
        "- LLM judge metrics (optional): Extra scoring by a judge model for grounding and relevance (costs tokens)."
    )

    st.markdown("### Performance metrics")
    st.markdown("- Median latency: Typical response time across questions.")
    st.markdown(
        "- P95 latency: Time within which 95% of responses completed (slow-case indicator)."
    )
    st.markdown("- Per-question latency: End-to-end response time for a single question.")

with st.expander("How can users improve results?", expanded=False):
    st.markdown(
        "- Ask specific, document-focused questions rather than very broad questions."
    )
    st.markdown(
        "- Avoid asking about information not present in the uploaded PDFs."
    )
    st.markdown(
        "- Use follow-up questions to narrow scope (e.g., “in section X…”, “for sandwich Y…”)."
    )
    st.markdown(
        "- Upload text-based PDFs where possible (scanned images may reduce retrieval quality)."
    )
    st.markdown(
        "- Use the Evaluate page to identify weak areas and iterate on your questions."
    )

with st.expander("Limitations & future improvements", expanded=False):
    st.markdown(
        "- Keyword-based relevance does not capture semantic equivalence (paraphrases/synonyms)."
    )
    st.markdown("- Small evaluation sets provide limited statistical confidence.")
    st.markdown("- Judge metrics increase cost and latency.")
    st.markdown(
        "- Future work: semantic similarity scoring, larger labelled evalsets, and human evaluation."
    )
