import pytest
from backend.state import AgentState
from backend.graph.engine import StateGraph

def test_state_graph_linear_transition():
    """Verify that StateGraph transitions through linear nodes properly mutating state."""
    graph = StateGraph()

    def step_one(state: AgentState) -> AgentState:
        return {"status": "retrieving", "raw_query": state["raw_query"].lower()}

    def step_two(state: AgentState) -> AgentState:
        return {"status": "qa_ready", "error_message": None}

    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.set_entry_point("step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", "END")

    runner = graph.compile()
    initial_state: AgentState = {
        "raw_query": "Attention Is All You Need",
        "query_type": "topic",
        "status": "init",
        "current_node": "",
        "mode": "mock",
        "candidate_papers": [],
        "parsed_sections": [],
        "chunks": [],
        "qa_history": [],
        "retrieval_available": False,
    }

    final_state = runner.run(initial_state)

    assert final_state["status"] == "qa_ready"
    assert final_state["raw_query"] == "attention is all you need"
    assert final_state["current_node"] == "step_two"


def test_state_graph_conditional_routing():
    """Verify that StateGraph correctly follows dynamic conditional edges."""
    graph = StateGraph()

    def parse_query(state: AgentState) -> AgentState:
        if state["raw_query"].startswith("1706."):
            return {"query_type": "paper_id"}
        return {"query_type": "topic"}

    def route_query(state: AgentState) -> str:
        if state["query_type"] == "paper_id":
            return "lookup_direct"
        return "search_topic"

    def lookup_direct(state: AgentState) -> AgentState:
        return {"status": "retrieving", "arxiv_id": state["raw_query"]}

    def search_topic(state: AgentState) -> AgentState:
        return {"status": "retrieving", "search_keywords": state["raw_query"]}

    graph.add_node("parse_query", parse_query)
    graph.add_node("lookup_direct", lookup_direct)
    graph.add_node("search_topic", search_topic)

    graph.set_entry_point("parse_query")
    graph.add_conditional_edge("parse_query", route_query)
    graph.add_edge("lookup_direct", "END")
    graph.add_edge("search_topic", "END")

    runner = graph.compile()

    state_id: AgentState = {
        "raw_query": "1706.03762",
        "status": "init",
        "current_node": "",
        "mode": "mock"
    }
    result_id = runner.run(state_id)
    assert result_id["query_type"] == "paper_id"
    assert result_id["arxiv_id"] == "1706.03762"
    assert result_id["current_node"] == "lookup_direct"

    state_topic: AgentState = {
        "raw_query": "Transformer attention mechanisms",
        "status": "init",
        "current_node": "",
        "mode": "mock"
    }
    result_topic = runner.run(state_topic)
    assert result_topic["query_type"] == "topic"
    assert result_topic["search_keywords"] == "Transformer attention mechanisms"
    assert result_topic["current_node"] == "search_topic"
