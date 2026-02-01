"""
Builds a FAISS index from the Gale Encyclopedia PDF.

Pipeline:
1. Validate config
2. Load PDF
3. Split text into chunks
4. Generate embeddings
5. Build FAISS index
6. Save index to disk

Run:
    python scripts/build_vectorstore.py
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.text_splitter import TextSplitter
from src.retrieval.vector_store import VectorStore
from src.utils.config import validate_config
from src.utils.logger import get_main_logger, log_section

logger = get_main_logger()


def build_vectorstore() -> bool:
    log_section(logger, "MEDIBOT VECTOR STORE BUILDER")

    try:
     
        # Step 0: Validate config
     
        logger.info("Validating configuration...")
        is_valid, errors = validate_config()
        if not is_valid:
            logger.error("Configuration validation failed:")
            for err in errors:
                logger.error(f"  - {err}")
            return False

       
        # Step 1: PDF ingestion
      
        logger.info("Step 1/4: Processing PDF")
        processor = PDFProcessor()
        documents = processor.process()

        if not documents:
            logger.error("PDFProcessor returned 0 documents/pages.")
            return False

        stats = processor.get_statistics(documents)
        logger.info(f"Processed {stats.get('total_pages', 'UNKNOWN')} pages from {stats.get('source_file', 'UNKNOWN')}")

       
        # Step 2: Text chunking
        
        logger.info("Step 2/4: Splitting text into chunks")
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        if not chunks:
            logger.error("TextSplitter returned 0 chunks. Check PDF extraction/cleaning.")
            return False

        logger.info(f"Created {len(chunks)} chunks")

       
        # Step 3: Embeddings + FAISS
        
        logger.info("Step 3/4: Creating vector store (this may take a while)")
        vector_store = VectorStore()
        vector_store.create_from_documents(chunks)

        vs_stats = vector_store.get_statistics() or {}
        total_vectors = vs_stats.get("total_vectors", None)
        if total_vectors is None:
            logger.warning("Vector store stats missing total_vectors (continuing).")
        else:
            logger.info(f"Built index with {total_vectors} vectors")

       
        # Step 4: Save to disk
       
        logger.info("Step 4/4: Saving vector store to disk")
        vector_store.save()
        logger.info(" Vector store saved successfully")

        
        # Summary
        
        log_section(logger, "BUILD COMPLETE")
        logger.info(f"Pages processed : {stats.get('total_pages', 'UNKNOWN')}")
        logger.info(f"Chunks created  : {len(chunks)}")
        logger.info(f"Vectors indexed : {vs_stats.get('total_vectors', 'UNKNOWN')}")
        logger.info(f"Index path      : {vs_stats.get('index_path', 'UNKNOWN')}")

        
        # Sanity check query (in-memory)
       
        logger.info("Running sanity test query (in-memory search)...")
        results = vector_store.search("What are the symptoms of diabetes?", k=2)
        logger.info(f"Sanity query retrieved {len(results)} results")

        print(" VECTOR STORE BUILD SUCCESSFUL")
        print("Next steps:")
        print("  1) chainlit run main.py")
        print("  2) Open http://localhost:8000")
        print("  3) Ask medical questions!\n")

        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(" ERROR: PDF file not found")
        print(f"Expected PDF location: {PROJECT_ROOT / 'data' / 'raw'}\n")
        return False

    except Exception as e:
        logger.exception("Vector store build failed")
        print(f" ERROR: {e}")
        return False


def main() -> None:
    print("\nMEDIBOT – VECTOR STORE BUILDER\n")
    success = build_vectorstore()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()