import os
from pathlib import Path
from typing import List, Tuple
import warnings

# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
LOGS_DIR = PROJECT_ROOT / "logs"
ASSETS_DIR = PROJECT_ROOT / "assets"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = PROJECT_ROOT / ".cache"

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VECTORSTORE_DIR,
    LOGS_DIR,
    CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# DEVICE CONFIGURATION (SAFE)

DEVICE = "cpu"
try:
    import torch

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
except Exception:
    warnings.warn("Torch not available. Falling back to CPU.")

# MODEL CONFIGURATION

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

LLM_MODEL_NAME = "google/flan-t5-base"
LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 512
LLM_TOP_P = 0.9
LLM_DO_SAMPLE = True

# DOCUMENT PROCESSING

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]

PDF_EXTRACTION_METHOD = "pypdf"
PRESERVE_FORMATTING = True

# EMBEDDING PERFORMANCE 

BATCH_SIZE = 32                 # Safe for Apple Silicon
ENABLE_EMBEDDING_CACHE = True   # Uses .cache directory

# VECTOR STORE

VECTORSTORE_INDEX_NAME = "medibot_faiss"
VECTORSTORE_INDEX_PATH = VECTORSTORE_DIR / VECTORSTORE_INDEX_NAME

SIMILARITY_SEARCH_K = 4
SIMILARITY_SCORE_THRESHOLD = 0.5
INDEX_TYPE = "Flat"

# PROMPTS

SYSTEM_PROMPT = """You are Medibot, a medical information assistant powered by the Gale Encyclopedia of Medicine.

Rules:
- Answer ONLY from provided context
- If not found, say you do not have the information
- Do NOT provide diagnoses or treatment
- Always advise consulting healthcare professionals
"""

QA_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Answer based ONLY on the context above:"""

WELCOME_MESSAGE = """# Welcome to Medibot!

Powered by the **Gale Encyclopedia of Medicine (2nd Edition)**.

Educational use only. Always consult a medical professional.

What would you like to learn today?
"""

# LOGGING

LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "medibot.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_LOG_LEVEL = "INFO"

# VALIDATION

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000
MIN_CHUNKS_FOR_INDEXING = 5

ERROR_NO_PDF = "No PDF found in data/raw/"
ERROR_NO_VECTORSTORE = "Vector store missing. Run build_vectorstore.py"

# HELPERS

def validate_config() -> Tuple[bool, List[str]]:
    errors = []

    if not MIN_CHUNK_SIZE <= CHUNK_SIZE <= MAX_CHUNK_SIZE:
        errors.append("Invalid CHUNK_SIZE")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return len(errors) == 0, errors


def get_pdf_files() -> List[Path]:
    return list(RAW_DATA_DIR.glob("*.pdf"))


def vectorstore_exists() -> bool:
    return VECTORSTORE_INDEX_PATH.exists()


def load_env_overrides() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)

            global LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE
            LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME)
            EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)
            CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", CHUNK_SIZE))
        except Exception:
            warnings.warn("dotenv not installed; skipping env overrides")


load_env_overrides()

valid, issues = validate_config()
if not valid:
    warnings.warn(f"Config issues detected: {issues}")
