import pytest
from backend.tools.vector_store import (
    chunk_text_by_words,
    build_chunks_from_sections,
    index_paper_chunks,
    retrieve_relevant_chunks
)
from backend.agent import run_digest_pipeline

def test_chunking_with_overlap():
    """Verify sliding window chunker respects chunk size and overlap."""
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)
    chunks = chunk_text_by_words(text, chunk_size=50, overlap=10)

    assert len(chunks) == 3
    # Check that adjacent chunks share overlapping words
    assert "word40" in chunks[0]
    assert "word40" in chunks[1]


def test_build_chunks_from_sections():
    """Verify metadata tagging (paper_id, section, page, seq) on chunks."""
    sections = [
        {"title": "Introduction", "page": 1, "text": "This is an introductory text about transformers."},
        {"title": "Architecture", "page": 2, "text": "The model architecture has multi-head attention."}
    ]
    chunks = build_chunks_from_sections(sections, paper_id="1706.03762")
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["section"] == "Introduction"
    assert chunks[0]["metadata"]["page"] == 1
    assert chunks[1]["metadata"]["section"] == "Architecture"
    assert chunks[1]["metadata"]["page"] == 2


def test_chroma_indexing_and_retrieval():
    """Verify ChromaDB upsert and cosine similarity retrieval with metadata."""
    paper_id = "test_paper_123"
    chunks = [
        {
            "chunk_id": f"{paper_id}_1",
            "text": "The encoder consists of a stack of 6 identical layers with multi-head self-attention.",
            "metadata": {"paper_id": paper_id, "section": "Model Architecture", "page": 3}
        },
        {
            "chunk_id": f"{paper_id}_2",
            "text": "Training was carried out on 8 NVIDIA P100 GPUs for 3.5 days.",
            "metadata": {"paper_id": paper_id, "section": "Training", "page": 7}
        }
    ]

    index_id = index_paper_chunks(chunks, paper_id=paper_id)
    assert index_id.startswith("paper_")

    # Query 1: should retrieve Architecture chunk
    results = retrieve_relevant_chunks("How many layers does the encoder have?", paper_id=paper_id, top_k=1)
    assert len(results) == 1
    assert "identical layers" in results[0]["text"]
    assert results[0]["metadata"]["section"] == "Model Architecture"

    # Query 2: should retrieve Training chunk
    results_gpu = retrieve_relevant_chunks("Which GPUs were used for training?", paper_id=paper_id, top_k=1)
    assert len(results_gpu) == 1
    assert "NVIDIA" in results_gpu[0]["text"]
    assert results_gpu[0]["metadata"]["section"] == "Training"


def test_pipeline_graph_through_vector_indexing():
    """Verify full state graph flows through: Query -> arXiv -> PDF -> Chunk & Index."""
    state = run_digest_pipeline("1706.03762")
    assert state["status"] == "summarizing"
    assert state["retrieval_available"] is True
    assert state["index_id"] is not None
    assert len(state["chunks"]) > 10
