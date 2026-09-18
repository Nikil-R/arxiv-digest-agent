"""
backend/agent.py - Main Backend Agent Orchestrator.
Builds the state graph pipeline and provides execution helpers.
"""

from typing import Optional, Callable
from backend.state import AgentState
from backend.config import LLM_PROVIDER
from backend.graph.engine import StateGraph
from backend.graph.nodes import (
    query_understanding_node,
    arxiv_retrieval_node
)

def create_digest_graph() -> StateGraph:
    """Constructs the explicit state graph for the arXiv pipeline."""
    graph = StateGraph()

    # Register Nodes
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("arxiv_retrieval", arxiv_retrieval_node)

    # Define Graph Transitions
    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "arxiv_retrieval")

    # Conditional check: if retrieval failed, go to END
    def check_retrieval_status(state: AgentState) -> str:
        if state.get("status") == "error":
            return "END"
        # In future milestones, next node is "pdf_parsing"
        return "END"

    graph.add_conditional_edge("arxiv_retrieval", check_retrieval_status)

    return graph


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
        "mode": mode or LLM_PROVIDER,
        "error_message": None
    }


def run_digest_pipeline(
    query: str,
    mode: Optional[str] = None,
    on_step: Optional[Callable[[str, AgentState], None]] = None
) -> AgentState:
    """
    Executes the autonomous agent graph for a query.
    Returns the final state containing metadata, candidate papers, and status.
    """
    graph = create_digest_graph()
    runner = graph.compile()
    initial_state = initialize_state(query, mode=mode)
    return runner.run(initial_state, on_node_complete=on_step)
