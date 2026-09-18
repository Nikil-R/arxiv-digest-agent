"""
backend/graph/nodes.py - Pipeline Graph Nodes.
Implements individual stateful node transformations:
- Node 1: query_understanding_node
- Node 2: arxiv_retrieval_node
- Node 3: pdf_fetch_parse_node
"""

from typing import Dict, Any
from pathlib import Path
from backend.state import AgentState
from backend.tools.arxiv_client import (
    extract_arxiv_id,
    fetch_paper_by_id,
    search_papers_by_topic
)
from backend.tools.pdf_parser import (
    download_pdf,
    extract_sections_from_pdf
)

def query_understanding_node(state: AgentState) -> AgentState:
    """
    Stage 1: Query Understanding.
    Parses intent: topic search vs. specific paper lookup.
    """
    raw_query = state.get("raw_query", "").strip()
    detected_id = extract_arxiv_id(raw_query)

    if detected_id:
        return {
            "query_type": "paper_id",
            "arxiv_id": detected_id,
            "search_keywords": None,
            "status": "retrieving",
            "error_message": None
        }
    else:
        return {
            "query_type": "topic",
            "arxiv_id": None,
            "search_keywords": raw_query,
            "status": "retrieving",
            "error_message": None
        }


def arxiv_retrieval_node(state: AgentState) -> AgentState:
    """
    Stage 2 & 3: arXiv Retrieval and Selection/Ranking.
    - If paper_id: Directly fetches paper metadata via official API.
    - If topic: Searches arXiv API, ranks results, and selects the top candidate.
    - Gracefully handles realistic failure cases: 0 results or invalid ID.
    """
    query_type = state.get("query_type", "topic")

    try:
        if query_type == "paper_id":
            arxiv_id = state.get("arxiv_id")
            if not arxiv_id:
                return {
                    "status": "error",
                    "error_message": "No valid arXiv ID extracted from query."
                }

            paper = fetch_paper_by_id(arxiv_id)
            if not paper:
                return {
                    "candidate_papers": [],
                    "selected_paper": None,
                    "status": "error",
                    "error_message": f"Paper with arXiv ID '{arxiv_id}' was not found on arXiv."
                }

            return {
                "candidate_papers": [paper],
                "selected_paper": paper,
                "status": "parsing",
                "error_message": None
            }

        else:
            # Topic search
            topic = state.get("search_keywords", state.get("raw_query", ""))
            candidates, selected = search_papers_by_topic(topic, max_results=5)

            if not candidates or not selected:
                return {
                    "candidate_papers": [],
                    "selected_paper": None,
                    "status": "error",
                    "error_message": f"No papers found matching topic '{topic}'. Please try refining your keywords."
                }

            return {
                "candidate_papers": candidates,
                "selected_paper": selected,
                "arxiv_id": selected.get("arxiv_id"),
                "status": "parsing",
                "error_message": None
            }

    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"arXiv retrieval failed: {str(exc)}"
        }


def pdf_fetch_parse_node(state: AgentState) -> AgentState:
    """
    Stage 4: Fetch & Parse PDF.
    Downloads the paper PDF and extracts structured sections page-by-page.
    Gracefully handles realistic failures (network failure, scanned/unparseable PDF)
    by falling back to abstract-only mode and alerting downstream QA.
    """
    paper = state.get("selected_paper")
    if not paper:
        return {
            "status": "error",
            "error_message": "Cannot parse PDF: No paper selected in state."
        }

    pdf_url = paper.get("pdf_url")
    arxiv_id = paper.get("arxiv_id", "unknown_id")

    if not pdf_url:
        return {
            "parsing_status": "fallback_abstract_only",
            "fallback_reason": "No PDF URL available in arXiv metadata.",
            "retrieval_available": False,
            "status": "indexing"
        }

    try:
        # Download with local caching
        pdf_path = download_pdf(pdf_url, arxiv_id)

        # Extract structured sections
        sections, full_text, success, reason = extract_sections_from_pdf(pdf_path)

        if not success:
            # Realistic Failure Handling: Scanned or empty PDF
            # Fall back gracefully to abstract without halting pipeline
            return {
                "pdf_path": str(pdf_path),
                "parsed_sections": [],
                "full_text": None,
                "parsing_status": "fallback_abstract_only",
                "fallback_reason": reason,
                "retrieval_available": False,
                "status": "indexing"
            }

        return {
            "pdf_path": str(pdf_path),
            "parsed_sections": sections,
            "full_text": full_text,
            "parsing_status": "success",
            "fallback_reason": None,
            "retrieval_available": True,
            "status": "indexing"
        }

    except Exception as exc:
        # Graceful fallback on network or download exceptions
        return {
            "pdf_path": None,
            "parsed_sections": [],
            "full_text": None,
            "parsing_status": "fallback_abstract_only",
            "fallback_reason": f"PDF download or read error: {str(exc)}",
            "retrieval_available": False,
            "status": "indexing"
        }
