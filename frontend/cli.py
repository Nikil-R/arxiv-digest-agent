"""
frontend/cli.py - Terminal / Command-Line Interface.
Adheres strictly to the assessment constraints:
'A frontend/UI beyond a basic CLI or notebook interface is out of scope.
 A CLI or simple script-based interaction is completely fine.'
"""
import sys
import argparse
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
    print("[*] Running State Graph: [Query Understanding] -> [arXiv Retrieval]...")

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
        print(f"\nAbstract:\n{paper.get('abstract')[:350]}...")
        print("-" * 75)
        print(f"[+] Pipeline status: {state.get('status').upper()} (Ready for PDF Parsing)")
    else:
        print("\n[-] No paper selected.")

if __name__ == "__main__":
    main()
