from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.logger import get_main_logger, PerformanceLogger


class LLMHandler:
    """
    Loads and runs a small instruction-tuned Seq2Seq model (default: flan-t5-base).

    Goals:
    - Stable on Apple Silicon (MPS)
    - Deterministic answers for QA
    - Avoid "instruction parroting" (repeating the prompt/rules)
    """

    def __init__(self) -> None:
        self.logger = get_main_logger()

        self.model_name = os.getenv("MEDIBOT_LLM_MODEL", "google/flan-t5-base")
        self.device = os.getenv("MEDIBOT_LLM_DEVICE", "mps")

        # Safe device fallback
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"

        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None

        self.logger.info("Initialized LLMHandler")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")

    def load_model(self):
        """Backward-compatible: returns self (model is internal)."""
        self.load()
        return self

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with PerformanceLogger(self.logger, "Loading LLM"):
            self.logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.logger.info("Loading model weights...")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            # Move to device
            self._model.to(self.device)
            self._model.eval()

            self.logger.info("LLM loaded successfully")

    def generate(self, prompt: str) -> str:
        """
        Generate an answer from a fully-formed prompt (already includes context + question).
        """
        self.load()
        assert self._model is not None and self._tokenizer is not None

        # Generation knobs (env-overridable)
        max_new_tokens = int(os.getenv("MEDIBOT_LLM_MAX_NEW_TOKENS", "256"))
        num_beams = int(os.getenv("MEDIBOT_LLM_NUM_BEAMS", "4"))

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move tensors
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with PerformanceLogger(self.logger, "Generating response"):
            # Deterministic decoding + anti-parroting settings
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=True,              # now valid because num_beams>1
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=1.0,
            )

        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Extra safety: if model starts echoing prompt-style lines, strip common artifacts
        # (kept minimal to avoid deleting real medical content)
        bad_prefixes = [
            "Use ONLY", "Do NOT", "CONTEXT:", "QUESTION:", "ANSWER:", "RULES:"
        ]
        for bp in bad_prefixes:
            if text.startswith(bp):
                # If it parrots, return fallback-safe response
                return "I am not certain based on the provided medical sources."

        return text
