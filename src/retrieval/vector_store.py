from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import faiss
from langchain_community.vectorstores.faiss import FAISS

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.utils.logger import get_main_logger, log_section, PerformanceLogger


def _set_safety_env() -> None:
    # Keep native libs calmer on macOS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _try_get_config() -> dict:
    """
    Best-effort config loader.
    Works with multiple project variants without forcing a specific API.
    """
    # Try get_config()
    try:
        from src.utils.config import get_config  # type: ignore
        cfg = get_config()
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass

    # Try load_config()
    try:
        from src.utils.config import load_config  # type: ignore
        cfg = load_config()
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass

    # Fallback defaults (aligned with your project structure)
    project_root = Path(__file__).resolve().parent.parent.parent
    return {
        "embeddings": {
            "model_name": os.getenv("MEDIBOT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "device": os.getenv("MEDIBOT_EMBED_DEVICE", "cpu"),
        },
        "vectorstore": {
            "index_path": os.getenv("MEDIBOT_INDEX_PATH", str(project_root / "vectorstore" / "medibot_faiss")),
        },
    }


class VectorStore:
    """
    Safe FAISS VectorStore wrapper for Medibot.

    IMPORTANT:
    - We avoid FAISS.load_local() (can segfault on some macOS setups).
    - We load index.faiss + index.pkl manually and reconstruct FAISS().
    """

    def __init__(self, index_path: Optional[str | Path] = None):
        self.logger = get_main_logger()
        cfg = _try_get_config()

        # Config values with safe fallbacks
        model_name = cfg.get("embeddings", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = os.getenv("MEDIBOT_EMBED_DEVICE", cfg.get("embeddings", {}).get("device", "cpu"))

        default_index_path = cfg.get("vectorstore", {}).get("index_path", "vectorstore/medibot_faiss")
        self.index_path = Path(index_path) if index_path else Path(default_index_path)

        self.embedding_generator = EmbeddingGenerator(model_name=model_name, device=device)
        self.vectorstore: Optional[FAISS] = None

        self.logger.info(f"Initialized VectorStore (index_path={self.index_path})")

    def create_from_documents(self, documents):
        log_section(self.logger, "VECTOR STORE CREATION")
        _set_safety_env()

        with PerformanceLogger(self.logger, f"Creating vector store from {len(documents)} documents"):
            embeddings = self.embedding_generator.load_model()
            self.vectorstore = FAISS.from_documents(documents, embeddings)

        self.logger.info(f"Vector store created with {self.vectorstore.index.ntotal} vectors")

    def save(self) -> None:
        if self.vectorstore is None:
            raise RuntimeError("Vector store is not created/loaded. Nothing to save.")

        _set_safety_env()
        self.index_path.mkdir(parents=True, exist_ok=True)

        with PerformanceLogger(self.logger, f"Saving vector store to {self.index_path}"):
            self.vectorstore.save_local(str(self.index_path))

        self.logger.info("Vector store saved successfully")

    def load(self) -> None:
        log_section(self.logger, "VECTOR STORE LOADING (SAFE)")
        _set_safety_env()

        index_file = self.index_path / "index.faiss"
        pkl_file = self.index_path / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():
            raise FileNotFoundError(
                f"Missing vectorstore files.\nExpected:\n  {index_file}\n  {pkl_file}"
            )

        # 1) Load embeddings (your safe SentenceTransformer wrapper)
        with PerformanceLogger(self.logger, "Loading embedding model"):
            embeddings = self.embedding_generator.load_model()

        # 2) Load FAISS index safely
        with PerformanceLogger(self.logger, f"Reading FAISS index: {index_file.name}"):
            index = faiss.read_index(str(index_file))

        # 3) Load docstore + index_to_docstore_id safely
        with PerformanceLogger(self.logger, f"Loading metadata: {pkl_file.name}"):
            with open(pkl_file, "rb") as f:
                obj = pickle.load(f)

        try:
            docstore, index_to_docstore_id = obj
        except Exception:
            raise RuntimeError(
                f"Unexpected format inside {pkl_file}. "
                f"Expected a tuple: (docstore, index_to_docstore_id)."
            )

        # 4) Construct FAISS VectorStore (positional args are version-safe)
        self.vectorstore = FAISS(embeddings, index, docstore, index_to_docstore_id)

        self.logger.info(f"✅ Vector store loaded. ntotal={self.vectorstore.index.ntotal}")

    def search(self, query: str, k: int = 3):
        if self.vectorstore is None:
            raise RuntimeError("Vector store not loaded. Call load() first.")
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 3):
        if self.vectorstore is None:
            raise RuntimeError("Vector store not loaded. Call load() first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_statistics(self) -> Dict[str, Any]:
        total = None
        try:
            if self.vectorstore is not None and hasattr(self.vectorstore, "index"):
                total = int(self.vectorstore.index.ntotal)
        except Exception:
            total = None
        return {
            "index_path": str(self.index_path),
            "total_vectors": total,
        }
