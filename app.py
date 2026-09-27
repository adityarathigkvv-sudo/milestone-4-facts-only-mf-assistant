import json
import os
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
with open(BASE / "data" / "corpus.json", "r", encoding="utf-8") as f:
    CORPUS = json.load(f)

SOURCE_MAP = {}
for item in CORPUS:
    SOURCE_MAP[item["source_id"]] = {
        "title": item["source_title"],
        "url": item["source_url"],
        "date": item["source_date"],
    }

documents = [
    f'{x["scheme"]} {x["topic"]} {x["text"]}'
    for x in CORPUS
]
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
MATRIX = VECTORIZER.fit_transform(documents)

REFUSAL_TERMS = [
    "should i", "should i buy", "should i invest", "should i sell",
    "buy or sell", "best fund", "which fund is best", "recommend",
    "recommendation", "portfolio", "allocate", "allocation", "returns",
    "return", "profit", "target", "outperform", "underperform", "good investment",
    "worth investing", "where should i invest", "which one should i choose"
]
PII_PATTERNS = [
    r"\b\d{10}\b",                       # common phone-like sequence
    r"\b[A-Z]{5}\d{4}[A-Z]\b",           # PAN
    r"\b\d{12}\b",                       # Aadhaar-like sequence
    r"\b\d{4,18}\b",                     # broad account/OTP guard
]
PII_WORDS = ["pan", "aadhaar", "aadhar", "otp", "one time password", "bank account", "account number", "phone number", "mobile number", "email address"]

def looks_like_pii(q: str) -> bool:
    ql = q.lower()
    if any(w in ql for w in PII_WORDS):
        return True
    return any(re.search(p, q, re.I) for p in PII_PATTERNS)

def is_advice(q: str) -> bool:
    ql = q.lower()
    return any(term in ql for term in REFUSAL_TERMS)

def retrieve(q: str, k: int = 3):
    scores = cosine_similarity(VECTORIZER.transform([q]), MATRIX)[0]
    idxs = scores.argsort()[::-1][:k]
    return [(CORPUS[i], float(scores[i])) for i in idxs]

def choose_fact(results, q):
    ql = q.lower()
    # Prefer topic matches before raw TF-IDF score.
    topic_order = [
        ("exit", "exit_load"),
        ("load", "exit_load"),
        ("sip", "minimum_sip"),
        ("minimum investment", "minimum_investment"),
        ("minimum", "minimum_investment"),
        ("lock", "lock_in"),
        ("elss", "lock_in"),
        ("benchmark", "benchmark"),
        ("riskometer", "riskometer"),
        ("risk", "riskometer"),
        ("statement", "capital_gains_statement"),
        ("capital gain", "capital_gains_statement"),
        ("cas", "cas"),
        ("category", "category"),
        ("what is", "category"),
    ]
    desired = None
    for phrase, topic in topic_order:
        if phrase in ql:
            desired = topic
            break
    if desired:
        for item, score in results:
            if item["topic"] == desired and score >= 0.05:
                return item, score
    return results[0] if results else (None, 0)

def fallback_source():
    s = next(x for x in SOURCE_MAP.values() if "Offer Document" in x["title"])
    return s

def answer(q):
    if not q.strip():
        return "Please enter a factual mutual-fund question."

    if looks_like_pii(q) or any(w in q.lower() for w in ["otp", "one time password"]):
        return ("For privacy, please do not enter PAN, Aadhaar, OTPs, bank/account numbers, "
                "phone numbers, or email addresses. This prototype does not accept or store PII. "
                "Last updated from sources: Current prototype privacy rule.")

    if is_advice(q):
        s = next(x for x in SOURCE_MAP.values() if "Introduction to Mutual Funds" in x["title"])
        return (f"I can provide facts such as expense/load, SIP minimums, lock-in, benchmark, riskometer, "
                f"and statement steps, but I can’t recommend buying, selling, holding, or choosing a fund. "
                f"Educational source: {s['url']} Last updated from sources: {s['date']}.")

    results = retrieve(q)
    item, score = choose_fact(results, q)

    if item is None or score < 0.10:
        s = fallback_source()
        return (f"I don’t have that fact in this prototype corpus, so I won’t guess. "
                f"Official SBI Mutual Fund documents: {s['url']}")

    s = SOURCE_MAP[item["source_id"]]
    # Keep response <= 3 sentences and include exactly one clear URL.
    return f"{item['text']} Source: {s['url']} Last updated from sources: {s['date']}."

st.set_page_config(page_title="Facts-Only MF Assistant", page_icon="📘", layout="centered")

st.title("📘 Facts-Only MF Assistant")
st.caption("RAG prototype • SBI Mutual Fund • factual Q&A only")

st.info(
    "Facts-only. No investment advice. Uses a small corpus of official SBI Mutual Fund, "
    "AMFI and SEBI public sources. Do not enter PAN, Aadhaar, OTPs, bank/account numbers, "
    "phone numbers or email addresses."
)

st.markdown("### Try an example")
examples = [
    "What is the exit load of SBI Large Cap Fund?",
    "What is the lock-in period of SBI ELSS Tax Saver Fund?",
    "How do I download my capital-gains statement?"
]
cols = st.columns(3)
for i, q in enumerate(examples):
    if cols[i].button(q, use_container_width=True):
        st.session_state["question"] = q

q = st.text_input(
    "Ask a factual question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is the benchmark of SBI Flexicap Fund?"
)

if st.button("Get factual answer", type="primary"):
    st.markdown("### Answer")
    st.write(answer(q))

with st.expander("How the RAG prototype works"):
    st.write(
        "1) The question is converted into a TF-IDF representation. "
        "2) Similar fact chunks from the official-source corpus are retrieved. "
        "3) A constrained answer template returns the retrieved fact with one source link. "
        "Advice and PII patterns are blocked before retrieval."
    )

with st.expander("Scope"):
    st.write(
        "Schemes: SBI Large Cap Fund, SBI Flexicap Fund, SBI ELSS Tax Saver Fund. "
        "Investor services: SBI Mutual Fund / AMFI. Regulatory context: SEBI."
    )
