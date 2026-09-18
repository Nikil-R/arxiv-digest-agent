import pytest
from backend.tools.arxiv_client import (
    extract_arxiv_id,
    fetch_paper_by_id,
    search_papers_by_topic,
    calculate_relevance_score
)
from backend.agent import run_digest_pipeline

def test_extract_arxiv_id():
    """Verify regex patterns for paper IDs, versioned IDs, and URLs."""
    assert extract_arxiv_id("1706.03762") == "1706.03762"
    assert extract_arxiv_id("1706.03762v5") == "1706.03762"
    assert extract_arxiv_id("https://arxiv.org/abs/2309.06180") == "2309.06180"
    assert extract_arxiv_id("https://arxiv.org/pdf/2309.06180v2.pdf") == "2309.06180"
    assert extract_arxiv_id("recent work on KV-cache compression for LLMs") is None
    assert extract_arxiv_id("deep learning architectures") is None


def test_relevance_score_calculation():
    """Verify transparent ranking formula prioritizes matching tokens and recency."""
    topic = "KV-cache compression"
    paper_relevant = {
        "title": "Efficient KV-cache compression for Long-Context LLMs",
        "abstract": "We introduce a compression technique for KV-cache.",
        "published": "2024-01-15T00:00:00Z"
    }
    paper_irrelevant = {
        "title": "Quantum Spin Liquids in 2D",
        "abstract": "Study of magnetic structures.",
        "published": "2018-05-10T00:00:00Z"
    }

    score_rel = calculate_relevance_score(topic, paper_relevant)
    score_irrel = calculate_relevance_score(topic, paper_irrelevant)
    assert score_rel > score_irrel
    assert score_rel > 0.5


def test_live_fetch_by_valid_id():
    """Verify live arXiv API retrieval for canonical Attention Is All You Need paper."""
    paper = fetch_paper_by_id("1706.03762")
    assert paper is not None
    assert "Attention Is All You Need" in paper["title"]
    assert "Vaswani" in "".join(paper["authors"])
    assert paper["arxiv_id"] == "1706.03762"
    assert paper["pdf_url"].endswith(".pdf")


def test_failure_handling_nonexistent_id():
    """Verify graceful handling when arXiv ID does not exist."""
    paper = fetch_paper_by_id("0000.00000")
    assert paper is None


def test_pipeline_graph_paper_id_flow():
    """Test full state graph execution with a paper ID."""
    state = run_digest_pipeline("1706.03762", mode="mock")
    assert state["query_type"] == "paper_id"
    assert state["selected_paper"] is not None
    assert "Attention Is All You Need" in state["selected_paper"]["title"]
    assert state["status"] in ["parsing", "indexing", "summarizing", "qa_ready"]
    assert state["error_message"] is None


def test_pipeline_graph_nonexistent_query():
    """Test realistic failure handling when query returns zero papers."""
    state = run_digest_pipeline("zyxwvutsrqpnonexistentquery9876543210", mode="mock")
    assert state["query_type"] == "topic"
    assert state["status"] == "error"
    assert state["selected_paper"] is None
    assert "No papers found matching topic" in state["error_message"]
