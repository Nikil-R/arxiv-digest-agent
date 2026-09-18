import pytest
from backend.agent import run_digest_pipeline, ask_question_state
from backend.tools.briefing_generator import generate_executive_briefing
from backend.tools.qa_engine import answer_paper_question

def test_executive_briefing_mock_mode():
    """Verify briefing contains all required sections from assessment rubric."""
    paper = {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "arxiv_id": "1706.03762",
        "published": "2017-06-12",
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
        "abstract": "The dominant sequence transduction models are based on complex recurrent networks."
    }
    briefing = generate_executive_briefing(paper, sections=[], mode="mock")
    content = briefing["content_markdown"]

    assert "Plain-English Summary" in content
    assert "Problem Statement" in content
    assert "Method / Approach" in content
    assert "Key Results & Claims" in content
    assert "Limitations" in content
    assert "Suggested Follow-up Questions" in content


def test_grounded_qa_answerable_question():
    """Verify QA mode returns grounded answer with citation for an answerable question."""
    paper = {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "abstract": "Transformer model."
    }
    answer, evidence = answer_paper_question(
        question="How many layers does the encoder stack have?",
        paper=paper,
        retrieval_available=True,
        mode="mock"
    )
    assert len(answer) > 20
    assert "Section:" in answer or len(evidence) > 0


def test_grounded_qa_unsupported_question():
    """
    Verify §2 & §5 Grounding and Anti-Hallucination:
    'If the answer isn't in the paper, the agent should say so rather than hallucinate.'
    """
    paper = {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "abstract": "Transformer model."
    }
    answer, evidence = answer_paper_question(
        question="What is the author's favorite movie?",
        paper=paper,
        retrieval_available=False,
        mode="mock"
    )
    assert "insufficient" in answer.lower() or "not provide" in answer.lower()


def test_pipeline_graph_through_full_briefing():
    """Verify state graph executes through to executive briefing and QA ready state."""
    state = run_digest_pipeline("1706.03762", mode="mock")
    assert state["status"] == "qa_ready"
    assert state["executive_briefing"] is not None

    # Test asking a follow up question
    state_after_qa = ask_question_state(state, "What is the Transformer architecture?")
    assert len(state_after_qa["qa_history"]) == 1
    assert state_after_qa["qa_history"][0]["question"] == "What is the Transformer architecture?"
    assert len(state_after_qa["qa_history"][0]["answer"]) > 20
