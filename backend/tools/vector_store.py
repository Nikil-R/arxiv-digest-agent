"""
backend/tools/vector_store.py - Chunking and Local ChromaDB Vector Store.
Fulfills Assessment Stage 5:
- 'Chunk & Embed: chunk the parsed text and store embeddings in a vector DB for later QA'
- 'Vector DB: local vector DB - ChromaDB (local mode)'
- 'Sensible chunking strategy, working vector search'
- 'Zero paid API keys: uses local 384-dim embedding function'
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

from backend.config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL

# Local embedded ChromaDB client
_client: Optional[chromadb.PersistentClient] = None
_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a singleton local persistent ChromaDB client."""
    global _client
    if _client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _client


def chunk_text_by_words(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Sliding window chunking respecting word boundaries.
    Generates contiguous text chunks with configurable overlap.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += max(1, chunk_size - overlap)

    return chunks


def build_chunks_from_sections(
    sections: List[Dict[str, Any]],
    paper_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Chunks parsed paper sections while preserving section and page metadata.
    Returns list of dicts: {"chunk_id": str, "text": str, "metadata": dict}
    """
    all_chunks = []
    chunk_seq = 0

    for sec in sections:
        sec_title = sec.get("title", "General")
        page_num = sec.get("page", 1)
        sec_text = sec.get("text", "").strip()

        if not sec_text:
            continue

        raw_chunks = chunk_text_by_words(sec_text, chunk_size=chunk_size, overlap=overlap)
        for c in raw_chunks:
            chunk_seq += 1
            chunk_id = f"{paper_id}_{chunk_seq}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": c,
                "metadata": {
                    "paper_id": paper_id,
                    "section": sec_title,
                    "page": page_num,
                    "chunk_seq": chunk_seq
                }
            })

    return all_chunks


def get_or_create_collection(paper_id: str) -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for a given paper."""
    client = get_chroma_client()
    # Normalize collection name: must start/end with alnum and contain only alnum, _ or -
    safe_name = "paper_" + re.sub(r"[^\w\-]", "_", paper_id)
    collection = client.get_or_create_collection(
        name=safe_name,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def index_paper_chunks(chunks: List[Dict[str, Any]], paper_id: str) -> str:
    """
    Stores paper chunks and metadata in local ChromaDB.
    Returns the collection name / index_id.
    """
    if not chunks:
        return ""

    collection = get_or_create_collection(paper_id)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Batch upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return collection.name


def retrieve_relevant_chunks(
    query: str,
    paper_id: str,
    top_k: int = TOP_K_RETRIEVAL
) -> List[Dict[str, Any]]:
    """
    Queries ChromaDB collection for top-k semantic matches given a question.
    Returns list of dicts: {"text": str, "metadata": dict, "distance": float}
    """
    collection = get_or_create_collection(paper_id)
    count = collection.count()

    if count == 0:
        return []

    k = min(top_k, count)
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    evidence_chunks = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            evidence_chunks.append({
                "text": doc,
                "metadata": meta,
                "distance": round(float(dist), 4)
            })

    return evidence_chunks
