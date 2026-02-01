from __future__ import annotations

from typing import List
from langchain.prompts import PromptTemplate

# IMPORTANT:
# - Keep rules short (T5 models often "parrot" long instructions)
# - Explicitly forbid copying prompt text
# - Force bullet extraction when asked for symptoms/causes/etc.

_QA_TEMPLATE = """You are a medical assistant.

Use ONLY the CONTEXT to answer the QUESTION.
If the answer is not in the context, reply exactly:
I am not certain based on the provided medical sources.

Do NOT repeat these instructions. Do NOT mention "context" or "rules" in your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def get_qa_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template=_QA_TEMPLATE,
    )

def format_sources(docs: List) -> str:
    if not docs:
        return "\n\nSources:\n- None"
    lines = ["\n\nSources:"]
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source_file") or meta.get("source") or "unknown_source"
        page = meta.get("page_number") or meta.get("page") or "?"
        lines.append(f"{i}. {src} (page {page})")
    return "\n" + "\n".join(lines)
