from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    MIN_CHUNK_SIZE,
    MIN_CHUNKS_FOR_INDEXING
)
from src.utils.logger import (
    get_ingestion_logger,
    PerformanceLogger,
    log_section
)

logger = get_ingestion_logger()


class TextSplitter:

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: List[str] = SEPARATORS
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )

        logger.info(
            f"Initialized TextSplitter (size={chunk_size}, overlap={chunk_overlap})"
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        log_section(logger, "TEXT SPLITTING")

        with PerformanceLogger(logger, "Splitting documents into chunks"):
            chunks = self.splitter.split_documents(documents)

            self._validate_chunks(chunks)
            self._add_chunk_metadata(chunks)
            self._log_chunk_statistics(chunks, documents)

            logger.info(" Text splitting complete")
            return chunks

    def _validate_chunks(self, chunks: List[Document]):
        if len(chunks) < MIN_CHUNKS_FOR_INDEXING:
            raise ValueError(
                f"Too few chunks generated ({len(chunks)}). "
                f"Minimum required: {MIN_CHUNKS_FOR_INDEXING}"
            )

        small_chunks = [c for c in chunks if len(c.page_content) < MIN_CHUNK_SIZE]
        if small_chunks:
            logger.warning(
                f"{len(small_chunks)} chunks smaller than {MIN_CHUNK_SIZE} chars"
            )

    def _add_chunk_metadata(self, chunks: List[Document]):
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "total_chunks": total
            })

    def _log_chunk_statistics(
        self,
        chunks: List[Document],
        original_docs: List[Document]
    ):
        total_original_chars = sum(len(d.page_content) for d in original_docs)
        total_chunk_chars = sum(len(c.page_content) for c in chunks)

        logger.info(f"Original documents: {len(original_docs)}")
        logger.info(f"Generated chunks: {len(chunks)}")
        logger.info(f"Original characters: {total_original_chars:,}")
        logger.info(f"Chunk characters: {total_chunk_chars:,}")
        logger.info(f"Average chunk size: {total_chunk_chars / len(chunks):.0f}")
        logger.info(f"Chunks per page: {len(chunks) / len(original_docs):.2f}")

    def get_chunk_preview(self, chunks: List[Document], n: int = 3) -> str:
        preview = []
        for i, chunk in enumerate(chunks[:n]):
            preview.append(f"Chunk {i + 1}")
            preview.append(f"  Length: {len(chunk.page_content)}")
            preview.append(f"  Text: {chunk.page_content[:200]}...")
            preview.append("")
        return "\n".join(preview)


def main():
    print("TESTING TEXT SPLITTER")

    sample_text = (
        "Diabetes mellitus is a chronic metabolic disorder characterized by "
        "high blood sugar levels. Treatment depends on the type of diabetes. "
    ) * 50

    documents = [
        Document(
            page_content=sample_text,
            metadata={"source": "test", "page": 1}
        )
    ]

    try:
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        print("SPLITTING RESULTS")
        print(f"Original docs: {len(documents)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"Overlap: {CHUNK_OVERLAP}")

        print("\nCHUNK PREVIEW\n")
        print(splitter.get_chunk_preview(chunks, n=3))

        print("\n Text splitter test successful!")
        print(f" Logs written to: {logger.handlers[1].baseFilename}\n")

    except Exception as e:
        print(f"\n Error: {e}")
        logger.exception("Text splitter test failed")


if __name__ == "__main__":
    main()
