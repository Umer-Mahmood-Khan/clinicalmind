"""
ingest.py — PDF Ingestion Pipeline for ClinicalMind
------------------------------------------------------
Loads PDFs from data/, splits them into chunks, embeds them with
OpenAI text-embedding-3-small, and persists a FAISS vector index
to vectorstore/.  A SHA-256 hash of every PDF is stored alongside
the index so the pipeline can skip rebuilding when nothing has changed.

Usage (standalone):
    python ingest.py
"""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # pulls OPENAI_API_KEY (and anything else) from .env

DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
HASH_FILE = VECTORSTORE_DIR / "doc_hashes.json"

# Chunk parameters chosen to balance context richness vs. noise
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # overlap keeps sentence boundaries intact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65_536), b""):
            sha.update(block)
    return sha.hexdigest()


def _current_hashes() -> dict[str, str]:
    """Compute hashes for every PDF currently in data/."""
    return {
        str(p): _hash_file(p)
        for p in sorted(DATA_DIR.glob("*.pdf"))
    }


def _saved_hashes() -> dict[str, str]:
    """Load previously saved hashes from disk (empty dict if file absent)."""
    if HASH_FILE.exists():
        with open(HASH_FILE) as fh:
            return json.load(fh)
    return {}


def _save_hashes(hashes: dict[str, str]) -> None:
    """Persist current hashes so the next run can compare."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(HASH_FILE, "w") as fh:
        json.dump(hashes, fh, indent=2)


def _needs_rebuild() -> bool:
    """
    Return True if the FAISS index is missing or any PDF has changed/been added/removed.
    """
    faiss_index = VECTORSTORE_DIR / "index.faiss"
    if not faiss_index.exists():
        return True  # no index yet

    return _current_hashes() != _saved_hashes()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_vectorstore() -> FAISS:
    """
    Load all PDFs → split → embed → save FAISS index.
    Returns the FAISS vectorstore object.
    """
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            "No PDF files found in data/. Add at least one PDF before ingesting."
        )

    print(f"[ingest] Loading {len(pdf_paths)} PDF(s)…")

    # --- 1. Load pages from every PDF ------------------------------------------
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Tag each page so agents can cite the original source file later
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        all_docs.extend(pages)
        print(f"  • {pdf_path.name}: {len(pages)} page(s)")

    print(f"[ingest] Total pages loaded: {len(all_docs)}")

    # --- 2. Split into manageable chunks ----------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split on paragraph breaks first, then sentences, then words, then chars
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[ingest] Split into {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # --- 3. Embed with OpenAI text-embedding-3-small ----------------------------
    # This model is cheap, fast, and well-suited for retrieval tasks
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    print("[ingest] Embedding chunks (this may take a moment)…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # --- 4. Persist to disk ------------------------------------------------------
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    _save_hashes(_current_hashes())
    print(f"[ingest] FAISS index saved to {VECTORSTORE_DIR}/")

    return vectorstore


# ---------------------------------------------------------------------------
# Public API consumed by other modules
# ---------------------------------------------------------------------------

def load_vectorstore() -> FAISS:
    """
    Return a ready-to-query FAISS vectorstore.

    If the index is stale (PDFs changed) or missing it is rebuilt first.
    Intended to be imported by agents.py and orchestrator.py.
    """
    if _needs_rebuild():
        print("[ingest] Index missing or documents changed — rebuilding…")
        return build_vectorstore()

    print("[ingest] Loading existing FAISS index from disk…")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we wrote this file ourselves
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_vectorstore()
    print("[ingest] Done.")
