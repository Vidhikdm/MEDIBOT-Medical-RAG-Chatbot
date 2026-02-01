from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils.config import RAW_DATA_DIR, ERROR_NO_PDF
from src.utils.logger import (
    get_ingestion_logger,
    PerformanceLogger,
    log_section,
)

logger = get_ingestion_logger()


class PDFProcessor:

    def __init__(self, pdf_path: Optional[Path] = None):
        self.pdf_path = pdf_path or self._find_pdf()
        logger.info(f"Initialized PDFProcessor with: {self.pdf_path.name}")

    def _find_pdf(self) -> Path:
        
        pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

        if not pdf_files:
            logger.error(ERROR_NO_PDF)
            raise FileNotFoundError(ERROR_NO_PDF)

        if len(pdf_files) > 1:
            logger.warning(
                f"Multiple PDFs found. Using first: {pdf_files[0].name}"
            )

        return pdf_files[0]

    def load_pdf(self) -> List[Document]:
      
        log_section(logger, "PDF LOADING")

        with PerformanceLogger(logger, f"Loading PDF: {self.pdf_path.name}"):
            loader = PyPDFLoader(str(self.pdf_path))
            documents = loader.load()

            logger.info(f"Pages loaded: {len(documents)}")
            logger.info(
                f"Total characters: {sum(len(d.page_content) for d in documents):,}"
            )

            if documents:
                logger.debug(
                    f"Sample text: {documents[0].page_content[:200]}..."
                )

            return documents

    def extract_metadata(self, documents: List[Document]) -> List[Document]:
        
        logger.info("Enhancing document metadata")

        total_pages = len(documents)

        for i, doc in enumerate(documents):
            doc.metadata.update(
                {
                    "source_file": self.pdf_path.name,
                    "page_number": i + 1,
                    "total_pages": total_pages,
                    "char_count": len(doc.page_content),
                    "source_type": "gale_encyclopedia",
                }
            )

        return documents

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text.
        """
        text = " ".join(text.split())
        text = text.replace("\x00", "")
        return text.strip()

    def process(self) -> List[Document]:
        
        documents = self.load_pdf()
        documents = self.extract_metadata(documents)

        logger.info("Cleaning text content")

        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)

        logger.info(" PDF processing completed")
        return documents

    def get_statistics(self, documents: List[Document]) -> dict:
       
        total_chars = sum(len(d.page_content) for d in documents)
        avg_chars = total_chars / len(documents) if documents else 0

        return {
            "source_file": self.pdf_path.name,
            "file_size_mb": self.pdf_path.stat().st_size / (1024 * 1024),
            "total_pages": len(documents),
            "total_characters": total_chars,
            "avg_characters_per_page": avg_chars,
        }


def main():
    #Test PDF processor.
    
    print("TESTING PDF PROCESSOR")

    try:
        processor = PDFProcessor()
        documents = processor.process()
        stats = processor.get_statistics(documents)

        print("PROCESSING RESULTS")
        print(f"Source File: {stats['source_file']}")
        print(f"File Size: {stats['file_size_mb']:.2f} MB")
        print(f"Total Pages: {stats['total_pages']}")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(
            f"Avg Characters/Page: {stats['avg_characters_per_page']:.0f}"
        )

        if documents:
            print("SAMPLE CONTENT (First 500 chars)")
            print(documents[0].page_content[:500])
            print("...")

        for handler in logger.handlers:
            if hasattr(handler, "baseFilename"):
                print(f"\n Logs written to: {handler.baseFilename}")
                break

        print("\n PDF processor test successful!")

    except FileNotFoundError as e:
        print(f"\n {e}")
        print(f"➡ Add PDF to: {RAW_DATA_DIR}")

    except Exception as e:
        logger.exception("PDF processing failed")
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
