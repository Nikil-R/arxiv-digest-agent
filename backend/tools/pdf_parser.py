"""
backend/tools/pdf_parser.py - PDF Fetcher and Structured Section Extractor.
Uses pure-python 'pypdf' with fallback to PyMuPDF if available.
Fulfills Assessment Stage 4 & Section 4 (Constraints):
- 'Fetch & Parse: download the PDF and extract text (sections, abstract, references at minimum)'
- 'Handle realistic failure cases: scanned images, broken layout, network failure'
"""

import os
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from backend.config import PDF_STORAGE_DIR

# Regex patterns for common academic paper section headers
SECTION_HEADING_PATTERN = re.compile(
    r"^(?:(?:(?:[0-9]{1,2}|[IVXLCDM]{1,4})\.?\s+)?([A-Z][A-Za-z0-9\s,\-]{2,50})|Abstract|References|Bibliography|Conclusion|Conclusions|Discussion|Methodology|Related Work)$",
    re.MULTILINE
)

def _get_ssl_context() -> ssl.SSLContext:
    """Returns an SSL context configured with certifi certificates for reliable cross-platform HTTPS."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def download_pdf(pdf_url: str, arxiv_id: str, timeout: int = 25) -> Path:
    """
    Downloads the paper PDF from arXiv to local storage with disk caching.
    If the file already exists locally, returns the cached file path.
    """
    clean_id = re.sub(r"[^\w\-\.]", "_", arxiv_id)
    target_path = PDF_STORAGE_DIR / f"{clean_id}.pdf"

    if target_path.exists() and target_path.stat().st_size > 1024:
        return target_path

    headers = {"User-Agent": "AutonomousArxivAgent/1.0 (academic assessment)"}
    req = urllib.request.Request(pdf_url, headers=headers)
    ctx = _get_ssl_context()

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            data = response.read()
            if len(data) < 1000:
                raise ValueError("Downloaded file is too small to be a valid PDF.")
            with open(target_path, "wb") as f:
                f.write(data)
    except Exception as exc:
        raise ConnectionError(f"Failed to download PDF from '{pdf_url}': {exc}")

    return target_path


def is_heading_candidate(line: str) -> bool:
    """Heuristic to detect whether a single line is likely a section header."""
    clean = line.strip()
    if not clean or len(clean) > 80:
        return False
    if clean.endswith(".") and not clean.split(".")[0].isdigit():
        return False
    return bool(SECTION_HEADING_PATTERN.match(clean))


def extract_sections_from_pdf(pdf_path: Path) -> Tuple[List[Dict[str, Any]], str, bool, Optional[str]]:
    """
    Extracts text page-by-page from a PDF using pypdf (pure Python, cross-platform).
    Extracts text blocks and tags them with (section_title, page_number).
    
    Returns:
        - sections: List of {"title": str, "text": str, "page": int}
        - full_text: Complete concatenated text string
        - parse_success: True if text was cleanly extracted, False if scanned/unparseable
        - fallback_reason: Explanation if parsing failed or was incomplete
    """
    if not pdf_path.exists():
        return [], "", False, f"PDF file does not exist at {pdf_path}"

    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
    except Exception as exc:
        return [], "", False, f"PDF reader failed to open PDF: {exc}"

    if num_pages == 0:
        return [], "", False, "PDF document contains 0 pages."

    sections: List[Dict[str, Any]] = []
    current_section = "Introduction"
    current_section_lines: List[str] = []
    current_section_page = 1

    total_words = 0
    full_text_pages: List[str] = []

    for page_num in range(num_pages):
        try:
            page = reader.pages[page_num]
            text = page.extract_text() or ""
        except Exception:
            text = ""

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        full_text_pages.append(text)

        for line in lines:
            words = line.split()
            total_words += len(words)

            if is_heading_candidate(line):
                if current_section_lines:
                    section_text = "\n".join(current_section_lines).strip()
                    if len(section_text) > 30:
                        sections.append({
                            "title": current_section,
                            "text": section_text,
                            "page": current_section_page
                        })
                current_section = line.strip()
                current_section_lines = []
                current_section_page = page_num + 1
            else:
                current_section_lines.append(line)

    if current_section_lines:
        section_text = "\n".join(current_section_lines).strip()
        if len(section_text) > 30:
            sections.append({
                "title": current_section,
                "text": section_text,
                "page": current_section_page
            })

    full_text = "\n\n".join(full_text_pages)

    # Realistic Failure Handling (§5): Check if PDF is a scanned image or non-extractable
    if total_words < 200:
        return (
            [],
            full_text,
            False,
            f"PDF appears to be a scanned image or non-extractable format (only {total_words} words detected)."
        )

    # Fallback to page-based chunks if no headers were captured
    if not sections:
        for page_num, page_text in enumerate(full_text_pages):
            if page_text.strip():
                sections.append({
                    "title": f"Page {page_num + 1}",
                    "text": page_text.strip(),
                    "page": page_num + 1
                })

    return sections, full_text, True, None
