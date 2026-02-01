#!/usr/bin/env python3
"""
MEDIBOT - QUICK RAG TEST (CORPUS-AWARE + CONTEXT-GROUNDED)

Run:
  MEDIBOT_EMBED_DEVICE=cpu python scripts/test_rag_quick.py

Pass criteria per query:
- vectorstore loads
- retrieval on-topic
- LLM answer is non-empty and not instruction-echo
- answer is GROUNDED in context (keywords from answer appear in retrieved context)
- sources attach
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def banner(title: str) -> None:
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)


def preview(text: str, n: int = 260) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


STOPWORDS = {
    "what","are","is","the","a","an","of","for","to","and","in","on","with","how",
    "symptoms","signs","causes","cause","treatment","treatments","diagnosis","diagnosed",
    "prevent","prevention","management","risk","risks","factors","factor","do","does",
    "can","could","should","when","why","list","explain","describe","common","usually",
    "often","may","might","include","including","patient","patients"
}

BAD_ECHO_PHRASES = [
    "use only the context",
    "do not repeat these instructions",
    "do not mention",
    "context:",
    "rules:",
]


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answer_looks_like_echo(ans: str) -> bool:
    a = normalize(ans)
    return any(p in a for p in BAD_ECHO_PHRASES)


def extract_topic(query: str) -> str:
    q = normalize(query)

    m = re.search(r"\b(symptoms|signs|causes|treatment|diagnosis|risk factors)\s+(of|for)\s+(.+)$", q)
    if m:
        return m.group(3).strip()

    m = re.search(r"\bwhat is\s+(.+)$", q)
    if m:
        return m.group(1).strip()

    m = re.search(r"\bhow is\s+(.+?)\s+(diagnosed|treated|managed)\b", q)
    if m:
        return m.group(1).strip()

    toks = [t for t in q.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(toks[:4]).strip() if toks else q


def topic_terms(topic: str) -> List[str]:
    t = normalize(topic)
    parts = [p for p in t.split() if p and p not in STOPWORDS]
    terms = []
    if t:
        terms.append(t)
    terms.extend(parts)
    # dedupe
    seen, out = set(), []
    for x in terms:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def doc_topic_score(text: str, topic: str) -> int:
    t = normalize(text)
    terms = topic_terms(topic)
    score = 0

    if terms and terms[0] in t:
        score += 12
    for term in terms[1:]:
        if term in t:
            score += 4
    if any(k in t for k in ["symptoms", "signs", "cause", "causes", "treatment", "diagnosis", "definition"]):
        score += 2
    if score == 0:
        score -= 20
    return score


def pick_best_query(vs, queries: List[str], topic: str, k: int) -> Tuple[str, List]:
    best_q, best_docs, best_total = "", [], -10**9
    for q in queries:
        docs = vs.search(q, k=k)
        scored = [(doc_topic_score(getattr(d, "page_content", ""), topic), d) for d in docs]
        kept = [d for sc, d in sorted(scored, key=lambda x: x[0], reverse=True) if sc >= 0]
        total = sum(max(0, sc) for sc, _ in scored)
        if total > best_total:
            best_total, best_q, best_docs = total, q, kept
    return best_q, best_docs


def trim_context(docs: List, topic: str, max_docs: int = 4, max_chars: int = 3200) -> Tuple[str, List]:
    scored = [(doc_topic_score(getattr(d, "page_content", ""), topic), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)

    kept, parts, used = [], [], 0
    for sc, d in scored:
        if sc < 0:
            continue
        text = (getattr(d, "page_content", "") or "").strip()
        if not text:
            continue
        if len(kept) >= max_docs:
            break
        if used + len(text) > max_chars and parts:
            break
        parts.append(text)
        kept.append(d)
        used += len(text)
    return "\n\n".join(parts), kept


def answer_keywords(ans: str) -> List[str]:
    a = normalize(ans)
    toks = [t for t in a.split() if t not in STOPWORDS and len(t) >= 4]
    # drop ultra-generic medical words that can false-match
    drop = {"disease","disorder","condition","medical","doctor","health"}
    toks = [t for t in toks if t not in drop]
    # dedupe, keep order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def grounded_score(ans: str, context: str) -> Tuple[int, List[str]]:
    """
    Score = how many answer keywords appear in context.
    Returns (hits, hit_terms)
    """
    c = normalize(context)
    kws = answer_keywords(ans)
    hit_terms = [k for k in kws if k in c]
    return len(hit_terms), hit_terms[:10]


def main() -> int:
    banner("MEDIBOT - QUICK RAG TEST (CORPUS-AWARE + CONTEXT-GROUNDED)")
    print("PWD           :", Path.cwd())
    print("Project root  :", PROJECT_ROOT)
    print("Python        :", sys.executable)
    print("PYTHONPATH(0) :", sys.path[0])
    print("EMBED_DEVICE  :", os.getenv("MEDIBOT_EMBED_DEVICE", "(not set)"))

    banner("STEP 1: Basic dependency imports")
    import numpy  # noqa
    import faiss  # noqa
    import pypdf  # noqa
    import torch  # noqa
    print(" numpy:", numpy.__version__)
    print(" faiss:", getattr(faiss, "__version__", "unknown"))
    print(" pypdf:", pypdf.__version__)
    print(" torch:", torch.__version__)
    print("mps available:", torch.backends.mps.is_available())

    banner("STEP 2: Project imports")
    try:
        from src.utils.config import validate_config
        from src.retrieval.vector_store import VectorStore
        from src.generation.llm_handler import LLMHandler
        from src.generation.prompt_templates import get_qa_prompt, format_sources
    except Exception as e:
        print("\n❌ Project imports failed:", e)
        return 1

    banner("STEP 3: Config validation")
    ok, errors = validate_config()
    print("Config valid:", ok)
    if not ok:
        print("Errors:", errors)
        return 1

    banner("STEP 4: VectorStore load test (disk)")
    vs = VectorStore()
    try:
        vs.load()
        print("✅ Vector store loaded from disk")
    except Exception as e:
        print("\n❌ Vector store load failed:", e)
        return 1

    # A–B corpus queries
    TEST_QUERIES = [
        "What are the symptoms of asthma?",
        "How is bronchitis diagnosed?",
        "What are common symptoms of appendicitis?",
    ]

    banner("STEP 5: Run test queries (retrieval must be on-topic + grounded)")
    llm = LLMHandler()

    for idx, query in enumerate(TEST_QUERIES, 1):
        print("\n" + "-" * 94)
        print(f"QUERY {idx}/{len(TEST_QUERIES)}: {query}")
        print("-" * 94)

        topic = extract_topic(query)
        print("Extracted topic:", topic)

        queries = [
            query,
            f"{topic}",
            f"{topic} symptoms",
            f"{topic} signs symptoms causes treatment",
            f"what is {topic}",
        ]

        best_q, docs = pick_best_query(vs, queries, topic, k=60)
        print("Best retrieval query:", best_q)
        print("Candidate docs kept:", len(docs))

        if not docs:
            print("\n❌ FAIL: No on-topic chunks retrieved for:", topic)
            return 1

        context, kept = trim_context(docs, topic, max_docs=4, max_chars=3200)
        print("Kept docs:", len(kept), "| Context chars:", len(context))

        banner(f"STEP 6.{idx}: Build prompt + generate")
        prompt = get_qa_prompt().format(context=context, question=query)
        print("Prompt length:", len(prompt))
        print("Prompt preview:", preview(prompt, 260))

        answer = (llm.generate(prompt) or "").strip()
        print("\nAnswer preview:", preview(answer, 260))
        print(format_sources(kept))

        if not answer:
            print("\n❌ FAIL: Empty answer.")
            return 1
        if answer_looks_like_echo(answer):
            print("\n❌ FAIL: Answer looks like instruction echo.")
            return 1

        hits, hit_terms = grounded_score(answer, context)
        print(f"\nGrounding check: {hits} keyword hits in context | hits={hit_terms}")

        # Strict but fair: require at least 2 keyword overlaps OR the model must say "not certain..."
        if "not certain based on the provided medical sources" not in normalize(answer):
            if hits < 2:
                print("\n❌ FAIL: Answer not grounded in retrieved context (prevents hallucination pass).")
                return 1

    banner("✅ QUICK RAG TEST PASSED (CONTEXT-GROUNDED)")
    print("Next step: chainlit run main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
