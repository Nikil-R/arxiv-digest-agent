"""
frontend/cli.py - Terminal / Command-Line Interface.
Adheres strictly to the assessment constraints:
'A frontend/UI beyond a basic CLI or notebook interface is out of scope.
 A CLI or simple script-based interaction is completely fine.'
"""
import sys
import argparse
from backend.agent import run_digest_pipeline, initialize_state

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
        choices=["llm", "mock"],
        default="mock",
        help="Pipeline mode: 'llm' for AI generation, 'mock' for offline deterministic test (default: mock)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("      Autonomous arXiv Paper Digest & QA Agent")
    print("=" * 70)

    query = args.query
    if not query:
        query = input("\nEnter research topic or arXiv ID/URL: ").strip()

    if not query:
        print("Error: No query provided. Exiting.")
        sys.exit(1)

    print(f"\n[+] Executing backend in mode: {args.mode.upper()}")
    print(f"[+] Query: {query}")

    state = run_digest_pipeline(query, mode=args.mode)
    print(f"[+] Pipeline status: {state['status']}")

if __name__ == "__main__":
    main()
