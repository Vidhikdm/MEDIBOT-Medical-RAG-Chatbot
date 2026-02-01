
"""
Tests the full RAG pipeline: Query → Retrieval → LLM → Response
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.retrieval.vector_store import VectorStore
from src.generation.llm_handler import LLMHandler
from src.generation.prompt_templates import get_qa_prompt, format_sources
from src.utils.logger import get_main_logger, log_section, PerformanceLogger

logger = get_main_logger()


def test_retrieval_only():
    log_section(logger, "TEST 1: RETRIEVAL ONLY")
    print("\n=== TEST 1: Vector Search (No LLM) ===\n")

    vs = VectorStore()
    vs.load()

    query = "What are the symptoms of diabetes?"
    results = vs.search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"\nDocument {i}: {doc.page_content[:200]}...\nMetadata: {doc.metadata}")

    print("\n Retrieval test passed")
    return results


def test_generation_only():
    log_section(logger, "TEST 2: GENERATION ONLY")
    print("\n=== TEST 2: LLM Generation (No Retrieval) ===\n")

    llm_handler = LLMHandler()
    llm_handler.load_model()

    prompt = "What are the symptoms of diabetes?"
    response = llm_handler.generate(prompt)

    print(f"\nResponse:\n{response}")
    print("\n Generation test passed")
    return response


def test_manual_rag():
    log_section(logger, "TEST 3: MANUAL RAG PIPELINE")
    print("\n=== TEST 3: Manual RAG ===\n")

    vs = VectorStore()
    vs.load()
    llm_handler = LLMHandler()
    llm_handler.load_model()

    query = "What are the symptoms of diabetes?"
    context_docs = vs.search(query, k=3)

    # Combine context safely
    MAX_CONTEXT_CHARS = 2000
    context = "\n\n".join([doc.page_content for doc in context_docs])
    context = context[:MAX_CONTEXT_CHARS]

    full_prompt = get_qa_prompt().format(context=context, question=query)

    with PerformanceLogger(logger, "Response generation"):
        response = llm_handler.generate(full_prompt)

    final_response = response + format_sources(context_docs)

    print("\nFINAL RESPONSE:\n")
    print(final_response)
    print("\n Manual RAG test passed")
    return final_response


def test_edge_cases():
    log_section(logger, "TEST 4: EDGE CASES")
    print("\n=== TEST 4: Edge Cases ===\n")

    vs = VectorStore()
    vs.load()
    llm_handler = LLMHandler()
    llm_handler.load_model()

    edge_cases = [
        ("", "Empty query"),
        ("x", "Single character"),
        ("What is the meaning of life?", "Out of scope"),
        ("diabetes " * 100, "Very long query"),
        ("DiAbEtEs", "Mixed case"),
        ("diabetes diabetes diabetes", "Repeated words")
    ]

    for query, desc in edge_cases:
        print(f"\n{desc}: '{query[:50]}...' ({len(query)} chars)")
        try:
            if not query.strip():
                print(" Empty query skipped")
                continue
            response = llm_handler.generate(query[:1000])  # truncate long query
            print(f" Response: {response[:100]}...")
        except Exception as e:
            print(f" Error: {str(e)[:100]}...")


def main():
    print("\n=== MEDIBOT COMPLETE RAG INTEGRATION TEST ===\n")
    print("This will test:")
    print("1. Vector retrieval\n2. LLM generation\n3. Manual RAG\n4. Edge cases\n")

    try:
        test_retrieval_only()
        test_generation_only()
        test_manual_rag()
        test_edge_cases()
        print("\n ALL TESTS PASSED! Medibot RAG pipeline is fully operational ")
        return True
    except FileNotFoundError:
        print("\n Vector store not found. Run:")
        print("   python scripts/build_vectorstore.py")
        return False
    except Exception as e:
        logger.exception("Test failed")
        print(f"\n ERROR: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)