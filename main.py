"""
Medibot - Main Application (Chainlit)
RAG medical Q&A chatbot.

Run:
  conda activate medibot
  MEDIBOT_EMBED_DEVICE=cpu chainlit run main.py
"""

from __future__ import annotations

import os
from typing import Optional, List, Any

import chainlit as cl

from src.retrieval.vector_store import VectorStore
from src.generation.llm_handler import LLMHandler
from src.generation.prompt_templates import get_qa_prompt, format_sources
from src.utils.logger import get_main_logger, log_section

logger = get_main_logger()

# ---- Safe config defaults (no ImportError risk) ----
WELCOME_MESSAGE = os.getenv(
    "MEDIBOT_WELCOME_MESSAGE",
    "👋 Hi! I'm Medibot.\nAsk me a medical question and I’ll answer using my indexed sources."
)

ERROR_NO_VECTORSTORE = os.getenv(
    "MEDIBOT_ERROR_NO_VECTORSTORE",
    "Vector store not found. Build it first using: `python scripts/build_vectorstore.py`"
)

# Toggle sources without touching config.py:
#   MEDIBOT_SHOW_SOURCES=1  -> show sources
#   MEDIBOT_SHOW_SOURCES=0  -> hide sources
SHOW_SOURCES = os.getenv("MEDIBOT_SHOW_SOURCES", "1").strip().lower() in {"1", "true", "yes", "y"}


# Globals (loaded once)
vector_store: Optional[VectorStore] = None
llm_handler: Optional[LLMHandler] = None


def _truncate(s: str, n: int = 2500) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 3] + "...")


def _build_prompt(question: str, docs: List[Any]) -> str:
    # Build CONTEXT string from retrieved docs
    context_parts: List[str] = []
    for d in docs or []:
        txt = getattr(d, "page_content", "") or ""
        txt = txt.strip()
        if txt:
            context_parts.append(txt)

    context = "\n\n".join(context_parts).strip()
    prompt = get_qa_prompt().format(context=context, question=question)
    return prompt


def _format_answer(answer: str, docs: List[Any]) -> str:
    ans = (answer or "").strip()
    if not ans:
        ans = "I am not certain based on the provided medical sources."

    out = _truncate(ans)

    if SHOW_SOURCES and docs:
        out += format_sources(docs)

    out += (
        "\n\n---\n"
        "*⚠️ Educational use only. Not medical advice. "
        "If you have symptoms or an emergency, consult a clinician or local emergency services.*"
    )
    return out


async def initialize_system() -> bool:
    global vector_store, llm_handler

    try:
        log_section(logger, "MEDIBOT INITIALIZATION")

        logger.info("Loading vector store...")
        vector_store = VectorStore()
        vector_store.load()
        logger.info("✅ Vector store loaded")

        logger.info("Loading LLM handler...")
        llm_handler = LLMHandler()

        # Some handlers need explicit load; some load during generate()
        try:
            llm_handler.load_model()
            logger.info("✅ LLM model loaded")
        except Exception as e:
            logger.warning(f"LLM load_model() skipped/failed (may still work): {e}")

        logger.info("✅ System initialized successfully")
        return True

    except FileNotFoundError:
        logger.error("Vector store not found")
        return False
    except Exception as e:
        logger.exception(f"Initialization failed: {e}")
        return False


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=WELCOME_MESSAGE).send()

    loading = cl.Message(content="🔄 Initializing Medibot...")
    await loading.send()

    ok = await initialize_system()
    if not ok:
        loading.content = (
            "❌ **Initialization Failed**\n\n"
            f"{ERROR_NO_VECTORSTORE}\n\n"
            "**Fix:**\n"
            "1. `cd ~/Developer/Medibot`\n"
            "2. `conda activate medibot`\n"
            "3. `python scripts/build_vectorstore.py`\n"
            "4. `chainlit run main.py`\n"
        )
        await loading.update()
        return

    loading.content = (
        "✅ **Medibot is ready!**\n\n"
        "- 📚 Knowledge base: *Gale Encyclopedia of Medicine (Vol 1: A–B)*\n"
        "- 🔍 Retrieval: FAISS vector search\n"
        "- 🧠 Generator: Local HF model via `LLMHandler`\n\n"
        "Ask a question (A–B topics work best)."
    )
    await loading.update()
    logger.info("Chat session started")


@cl.on_message
async def on_message(message: cl.Message):
    global vector_store, llm_handler

    if vector_store is None or llm_handler is None:
        await cl.Message(content="❌ System not initialized. Refresh the page.").send()
        return

    user_query = (message.content or "").strip()
    if not user_query:
        await cl.Message(content="Please type a question.").send()
        return

    thinking = cl.Message(content="🤔 Searching sources and generating an answer...")
    await thinking.send()

    try:
        logger.info(f"User query: {user_query}")

        docs = vector_store.search(user_query, k=3)
        if not docs:
            thinking.content = _format_answer("I am not certain based on the provided medical sources.", [])
            await thinking.update()
            return

        prompt = _build_prompt(user_query, docs)
        logger.info(f"Prompt chars: {len(prompt)}")

        answer = llm_handler.generate(prompt)

        thinking.content = _format_answer(answer, docs)
        await thinking.update()

    except Exception as e:
        logger.exception("Error processing message")
        thinking.content = (
            "❌ **Error Processing Query**\n\n"
            f"```\n{e}\n```\n\n"
            "**Try:**\n"
            "- Rephrasing your question\n"
            "- Asking about topics likely covered in the A–B encyclopedia volume\n"
        )
        await thinking.update()


@cl.on_chat_end
def on_chat_end():
    logger.info("Chat session ended")


if __name__ == "__main__":
    print("=" * 70)
    print("MEDIBOT - Medical Question Answering System (Chainlit)")
    print("=" * 70)
    print("Run:  conda activate medibot")
    print("      MEDIBOT_EMBED_DEVICE=cpu chainlit run main.py")
    print("Web:  http://localhost:8000")
    print("=" * 70)
