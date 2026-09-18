"""
backend/tools/briefing_generator.py - Structured Executive Briefing Generator.
Fulfills Assessment Section 2 (Output: Executive Briefing):
- Title, authors, arXiv ID, publish date, link
- 1-paragraph plain-English summary ("why this paper matters")
- Problem statement
- Method/approach (bullet points)
- Key results / claims
- Limitations (explicit - don't let the model skip this)
- Suggested follow-up questions a reader might ask
"""

from typing import Dict, Any, Optional
from backend.tools.llm_client import call_llm

BRIEFING_SYSTEM_PROMPT = """You are an expert scientific researcher and technical briefing specialist.
Your task is to analyze an academic research paper and generate a rigorous, structured Executive Briefing.
You must adhere strictly to the following markdown headings and format:

### Plain-English Summary
(1 paragraph explaining in clear language why this paper matters and its practical significance)

### Problem Statement
(Clear definition of the fundamental challenge, bottleneck, or open question the authors tackle)

### Method / Approach
(Bullet points detailing the core technical innovation, architectural choices, and mechanisms)

### Key Results & Claims
(Concrete quantitative or empirical findings, benchmark outcomes, and verified claims)

### Limitations
(Explicit discussion of constraints, computational overhead, assumptions, or failure modes. DO NOT skip or soften this section)

### Suggested Follow-up Questions
(3 insightful technical questions a reader or researcher might ask to probe deeper)
"""

def generate_executive_briefing(paper: Dict[str, Any], sections: list, mode: str = "groq") -> Dict[str, Any]:
    """
    Generates a structured executive briefing adhering to assessment requirements.
    Uses paper metadata and representative section excerpts.
    """
    title = paper.get("title", "Untitled")
    authors = ", ".join(paper.get("authors", []))
    arxiv_id = paper.get("arxiv_id", "")
    published = paper.get("published", "")
    pdf_url = paper.get("pdf_url", "")
    abstract = paper.get("abstract", "")

    # Build context from abstract and key section snippets
    context_parts = [f"Title: {title}", f"Authors: {authors}", f"Abstract: {abstract}"]
    for sec in sections[:6]:
        sec_title = sec.get("title", "")
        sec_text = sec.get("text", "")[:600]
        context_parts.append(f"--- Section: {sec_title} (Page {sec.get('page', 1)}) ---\n{sec_text}")

    context = "\n\n".join(context_parts)
    prompt = f"Analyze the following research paper and generate the structured Executive Briefing:\n\n{context}"

    analysis_markdown = call_llm(prompt, system_prompt=BRIEFING_SYSTEM_PROMPT, mode=mode)

    briefing_artifact = {
        "metadata": {
            "title": title,
            "authors": authors,
            "arxiv_id": arxiv_id,
            "published": published,
            "pdf_url": pdf_url
        },
        "content_markdown": analysis_markdown
    }

    return briefing_artifact
