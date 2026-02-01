from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

try:
    from langchain_core.embeddings import Embeddings
except Exception:
    from langchain.embeddings.base import Embeddings

from src.utils.logger import get_main_logger


def _safe_device(requested: Optional[str]) -> str:
    env = os.getenv("MEDIBOT_EMBED_DEVICE")
    if env:
        requested = env.strip().lower()
    if requested in {"mps", "cpu"}:
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_safety_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


@dataclass
class STEmbeddings(Embeddings):
    model: SentenceTransformer
    batch_size: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.logger = get_main_logger()
        self.model_name = model_name
        self.device = _safe_device(device)
        self._embeddings: Optional[STEmbeddings] = None

        self.logger.info("Initializing EmbeddingGenerator")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")

    def load(self) -> STEmbeddings:
        if self._embeddings is not None:
            return self._embeddings

        _set_safety_env()

        bs_env = os.getenv("MEDIBOT_EMBED_BATCH_SIZE", "32")
        try:
            batch_size = max(1, int(bs_env))
        except Exception:
            batch_size = 32

        model = SentenceTransformer(self.model_name, device=self.device)
        self._embeddings = STEmbeddings(model=model, batch_size=batch_size)
        return self._embeddings

    # backward-compatible alias
    def load_model(self) -> STEmbeddings:
        return self.load()

    @property
    def embeddings(self) -> STEmbeddings:
        return self.load()
