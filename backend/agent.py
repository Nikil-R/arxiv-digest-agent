"""
backend/agent.py - Main Backend Agent Orchestrator.
Exposes high-level methods to run the arXiv digest pipeline and QA queries.
"""
from typing import Optional, Callable
from backend.state import AgentState
from backend.config import LLM_PROVIDER
from backend.graph.engine import StateGraph

def initialize_state(query: str, mode: Optional[str] = None) -> AgentState:
    """Initializes a fresh AgentState dictionary for a given user query."""
    return {
        "raw_query": query.strip(),
        "query_type": "topic",
        "candidate_papers": [],
        "selected_paper": None,
        "parsed_sections": [],
        "chunks": [],
        "retrieval_available": False,
        "qa_history": [],
        "status": "init",
        "current_node": "",
        "mode": mode or LLM_PROVIDER
    }

def run_digest_pipeline(
    query: str,
    mode: Optional[str] = None,
    on_step: Optional[Callable[[str, AgentState], None]] = None
) -> AgentState:
    """
    Executes the autonomous agent graph for a query up to the executive briefing.
    Returns the updated state containing paper metadata, parsed content, and briefing.
    """
    # Graph instantiation and compilation will expand across milestones
    graph = StateGraph()
    state = initialize_state(query, mode=mode)
    return state
