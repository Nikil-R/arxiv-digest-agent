"""
backend/graph/nodes.py - Pipeline Graph Nodes.
Implements individual stateful node transformations:
- Node 1: query_understanding_node
- Node 2: arxiv_retrieval_node
"""

from typing import Dict, Any
from backend.state import AgentState
from backend.tools.arxiv_client import (
    extract_arxiv_id,
    fetch_paper_by_id,
    search_papers_by_topic
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
    mode = state.get("mode", "groq")

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
