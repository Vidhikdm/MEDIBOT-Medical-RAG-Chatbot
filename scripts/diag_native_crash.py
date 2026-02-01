#!/usr/bin/env python3
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def banner(t):
    print("\n" + "="*80)
    print(t)
    print("="*80)

banner("ENV")
print("Python:", sys.executable)
print("PWD:", os.getcwd())
print("Project root:", PROJECT_ROOT)

banner("1) TORCH IMPORT")
import torch
print("Torch version:", torch.__version__)
print("Torch file   :", torch.__file__)
print("MPS available:", torch.backends.mps.is_available())

banner("2) FAISS IMPORT + READ INDEX FILE")
import faiss
print("faiss version:", getattr(faiss, "__version__", "n/a"))

# Try to locate index.faiss
possible = [
    PROJECT_ROOT / "vectorstore" / "medibot_faiss" / "index.faiss",
    PROJECT_ROOT / "vectorstore" / "medibot_faiss" / "index" / "index.faiss",
    PROJECT_ROOT / "vectorstore" / "medibot_faiss.index" / "index.faiss",
]
index_path = None
for p in possible:
    if p.exists():
        index_path = p
        break

print("Detected index.faiss:", index_path)
if index_path:
    idx = faiss.read_index(str(index_path))
    print("Index loaded. ntotal:", idx.ntotal)
else:
    print("WARNING: Could not find index.faiss automatically. Check your vectorstore folder layout.")

banner("3) SENTENCE-TRANSFORMERS MODEL LOAD (CPU FIRST)")
from sentence_transformers import SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
m_cpu = SentenceTransformer(model_name, device="cpu")
print("Loaded on CPU OK:", model_name)

banner("4) SENTENCE-TRANSFORMERS MODEL LOAD (MPS)")
m_mps = SentenceTransformer(model_name, device="mps")
print("Loaded on MPS OK:", model_name)

banner("✅ DIAG COMPLETE - NO NATIVE CRASHES DETECTED")
