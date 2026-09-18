"""
backend/agent.py - Main Backend Agent Orchestrator.
Builds the full 7-stage state graph and exposes pipeline + QA methods.
"""

from typing import Optional, Callable, Dict, Any
from backend.state import AgentState
from backend.config import LLM_PROVIDER
from backend.graph.engine import StateGraph
from backend.graph.nodes import (
    query_understanding_node,
    arxiv_retrieval_node,
    pdf_fetch_parse_node,
    chunk_and_embed_node,
    summarize_node,
    qa_node
)

def create_digest_graph() -> StateGraph:
    """Constructs the explicit state graph for Stages 1 through 6."""
    graph = StateGraph()

    # Register Nodes
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("arxiv_retrieval", arxiv_retrieval_node)
    graph.add_node("pdf_fetch_parse", pdf_fetch_parse_node)
    graph.add_node("chunk_and_embed", chunk_and_embed_node)
    graph.add_node("summarize", summarize_node)

    # Define Transitions
    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "arxiv_retrieval")

    # Routing from retrieval: if error -> END, else -> pdf_fetch_parse
    def check_retrieval_status(state: AgentState) -> str:
        if state.get("status") == "error":
            return "END"
        return "pdf_fetch_parse"

    graph.add_conditional_edge("arxiv_retrieval", check_retrieval_status)
    graph.add_edge("pdf_fetch_parse", "chunk_and_embed")
    graph.add_edge("chunk_and_embed", "summarize")
    graph.add_edge("summarize", "END")

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
        "error_message": None,
        "parsing_status": "success",
        "fallback_reason": None,
        "index_id": None,
        "executive_briefing": None,
        "current_question": None,
        "retrieved_chunks": []
    }


def run_digest_pipeline(
    query: str,
    mode: Optional[str] = None,
    on_step: Optional[Callable[[str, AgentState], None]] = None
) -> AgentState:
    """Executes the full agent graph up to executive briefing generation."""
    graph = create_digest_graph()
    runner = graph.compile()
    initial_state = initialize_state(query, mode=mode)
    return runner.run(initial_state, on_node_complete=on_step)


def ask_question_state(state: AgentState, question: str) -> AgentState:
    """Executes Stage 7: Grounded Question-Answering over the current paper state."""
    state_copy = state.copy()
    state_copy["current_question"] = question.strip()
    return qa_node(state_copy)
