"""
backend/tools/qa_engine.py - Grounded RAG Question-Answering Engine.
Fulfills Assessment Section 2 (QA Mode) & Section 5 (Grounding & Anti-Hallucination):
- 'Answers must be grounded in retrieved chunks (RAG)'
- 'If the answer isn't in the paper, the agent should say so rather than hallucinate'
- 'Cites source sections and page numbers'
"""

from typing import Dict, Any, List, Tuple
from backend.tools.vector_store import retrieve_relevant_chunks
from backend.tools.llm_client import call_llm

QA_SYSTEM_PROMPT = """You are a grounded scientific paper question-answering assistant.
Your answers MUST be strictly based ONLY on the retrieved paper excerpts provided below.
Rules:
1. If the provided excerpts do not contain sufficient evidence to answer the question, you MUST explicitly state:
   "Based on the provided sections of this paper, there is insufficient evidence to answer this question."
2. Never extrapolate, speculate, or introduce outside knowledge not present in the excerpts.
3. At the end of your answer, you must cite the specific source sections and pages provided in the excerpts (e.g., [Section: Architecture, Page: 3]).
"""

def answer_paper_question(
    question: str,
    paper: Dict[str, Any],
    retrieval_available: bool,
    mode: str = "groq"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Answers a follow-up question strictly grounded in retrieved paper chunks.
    Returns:
        - answer_text: The grounded answer string with citations or explicit refusal.
        - evidence_chunks: The retrieved chunks used as evidence.
    """
    arxiv_id = paper.get("arxiv_id", "")
    abstract = paper.get("abstract", "")

    # Fallback Case: If PDF was unparseable/scanned and vector retrieval is unavailable
    if not retrieval_available:
        evidence = [{"text": abstract, "metadata": {"section": "Abstract", "page": 1}, "distance": 0.0}]
        prompt = f"""ANSWER THE USER QUESTION using only the paper's abstract and metadata:
Abstract: {abstract}

User Question: {question}

Note: Full PDF text was unavailable. Answer only if the abstract directly covers the question; otherwise state that the full paper text is required."""
        answer = call_llm(prompt, system_prompt=QA_SYSTEM_PROMPT, mode=mode)
        return answer, evidence

    # Normal RAG Case: Retrieve top-k chunks from ChromaDB
    retrieved = retrieve_relevant_chunks(question, paper_id=arxiv_id, top_k=4)

    if not retrieved:
        return "Based on the provided sections of this paper, there is insufficient evidence to answer this question.", []

    # Format retrieved evidence with citations
    context_blocks = []
    for i, item in enumerate(retrieved, 1):
        meta = item.get("metadata", {})
        sec = meta.get("section", "Unknown Section")
        page = meta.get("page", "?")
        context_blocks.append(f"[Excerpt {i} | Section: {sec}, Page {page}]\n{item['text']}")

    context_str = "\n\n".join(context_blocks)
    prompt = f"""ANSWER THE USER QUESTION using ONLY the following retrieved excerpts from the paper:

{context_str}

User Question: {question}"""

    answer = call_llm(prompt, system_prompt=QA_SYSTEM_PROMPT, mode=mode)
    return answer, retrieved
