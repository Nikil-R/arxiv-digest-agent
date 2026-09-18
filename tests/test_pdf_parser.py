import pytest
from pathlib import Path
import pypdf
from backend.tools.pdf_parser import (
    is_heading_candidate,
    extract_sections_from_pdf,
    download_pdf
)
from backend.agent import run_digest_pipeline

def test_heading_detection():
    """Verify regex and heuristic heading recognition."""
    assert is_heading_candidate("1 Introduction") is True
    assert is_heading_candidate("3. Model Architecture") is True
    assert is_heading_candidate("References") is True
    assert is_heading_candidate("Conclusion") is True
    assert is_heading_candidate("This is a standard body sentence that should not be a heading.") is False
    assert is_heading_candidate("x + y = z.") is False


def test_scanned_pdf_failure_handling(tmp_path):
    """
    Verify §5 realistic failure case:
    If a PDF has no extractable text (e.g., scanned images),
    system returns parse_success=False with an informative fallback reason.
    """
    fake_pdf = tmp_path / "scanned_doc.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(fake_pdf, "wb") as f:
        writer.write(f)

    sections, full_text, success, reason = extract_sections_from_pdf(fake_pdf)
    assert success is False
    assert "scanned image" in reason.lower()
    assert len(sections) == 0


def test_live_pdf_download_and_parse():
    """
    Test real downloading and section extraction on Attention Is All You Need (1706.03762).
    Verifies pages, headings, and full text extraction.
    """
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
    pdf_path = download_pdf(pdf_url, "1706.03762")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 10000

    sections, full_text, success, reason = extract_sections_from_pdf(pdf_path)
    assert success is True
    assert reason is None
    assert len(sections) > 0
    assert len(full_text) > 5000

    # Verify key sections exist
    section_titles = [s["title"].lower() for s in sections]
    assert any("introduction" in t for t in section_titles)
    assert any("reference" in t or "model" in t or "conclusion" in t for t in section_titles)


def test_pipeline_graph_with_pdf_parsing():
    """Verify the state graph executes through Query -> arXiv -> PDF Parsing -> Indexing."""
    state = run_digest_pipeline("1706.03762")
    assert state["status"] in ["indexing", "summarizing"]
    assert state["retrieval_available"] is True
    assert state["parsing_status"] == "success"
    assert state["pdf_path"] is not None
    assert len(state["parsed_sections"]) > 0
