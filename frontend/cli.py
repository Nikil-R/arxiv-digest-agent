"""
frontend/cli.py - Terminal / Command-Line Interface.
Adheres strictly to the assessment constraints:
'A frontend/UI beyond a basic CLI or notebook interface is out of scope.
 A CLI or simple script-based interaction is completely fine.'
"""
import sys
import io
import argparse

# Ensure standard output can safely display Unicode scientific symbols on Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from backend.agent import run_digest_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous arXiv Paper Digest & QA Agent (CLI Client)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="arXiv paper ID (e.g., '1706.03762'), arXiv URL, or research topic"
    )
    parser.add_argument(
        "--mode",
        choices=["groq", "gemini", "mock"],
        default="groq",
        help="Pipeline mode: 'groq' (primary), 'gemini' (fallback), 'mock' (offline) (default: groq)"
    )

    args = parser.parse_args()

    print("=" * 75)
    print("         Autonomous arXiv Paper Digest & QA Agent (8byte)")
    print("=" * 75)

    query = args.query
    if not query:
        query = input("\nEnter research topic or arXiv ID/URL: ").strip()

    if not query:
        print("[-] Error: No query provided. Exiting.")
        sys.exit(1)

    print(f"\n[+] Input Query: {query}")
    print(f"[+] Active LLM Mode: {args.mode.upper()}")
    print("[*] Running State Graph: [Query Understanding] -> [arXiv Retrieval] -> [PDF Fetch & Parse]...")

    state = run_digest_pipeline(query, mode=args.mode)

    if state.get("status") == "error":
        print(f"\n[-] Pipeline Error: {state.get('error_message')}")
        sys.exit(1)

    paper = state.get("selected_paper")
    if paper:
        print("\n" + "-" * 75)
        print("                  SELECTED PAPER DETAILS")
        print("-" * 75)
        print(f"Title:         {paper.get('title')}")
        print(f"Authors:       {', '.join(paper.get('authors', []))}")
        print(f"arXiv ID:      {paper.get('arxiv_id')}")
        print(f"Published:     {paper.get('published')}")
        print(f"Primary Cat:   {paper.get('primary_category')}")
        print(f"PDF Link:      {paper.get('pdf_url')}")
        if "relevance_score" in paper:
            print(f"Rank Score:    {paper.get('relevance_score')} (ranked top among candidates)")
        
        print("\n" + "-" * 75)
        print("                  PDF EXTRACTION SUMMARY")
        print("-" * 75)
        print(f"Local PDF:     {state.get('pdf_path')}")
        print(f"Parse Status:  {state.get('parsing_status').upper()}")
        print(f"Retrieval OK:  {state.get('retrieval_available')}")
        if state.get("fallback_reason"):
            print(f"Note:          {state.get('fallback_reason')}")
        
        sections = state.get("parsed_sections", [])
        print(f"Extracted:     {len(sections)} sections/blocks detected")
        if sections:
            print("\nSample Extracted Sections:")
            for s in sections[:4]:
                snippet = s['text'][:90].replace('\n', ' ')
                print(f"  - [{s['title']}] (Page {s['page']}): \"{snippet}...\"")
                
        print("-" * 75)
        print(f"[+] Pipeline status: {state.get('status').upper()} (Ready for Chunking & Embedding)")
    else:
        print("\n[-] No paper selected.")

if __name__ == "__main__":
    main()
