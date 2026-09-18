"""
frontend/cli.py - Terminal / Command-Line Interface.
Adheres strictly to the assessment constraints:
'A frontend/UI beyond a basic CLI or notebook interface is out of scope.
 A CLI or simple script-based interaction is completely fine.'
"""
import sys
import io
import argparse

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from backend.agent import run_digest_pipeline, ask_question_state

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
    print("[*] Running State Graph: [Query] -> [arXiv] -> [PDF Parse] -> [Index] -> [Briefing]...")

    state = run_digest_pipeline(query, mode=args.mode)

    if state.get("status") == "error":
        print(f"\n[-] Pipeline Error: {state.get('error_message')}")
        sys.exit(1)

    paper = state.get("selected_paper")
    briefing = state.get("executive_briefing")

    if not paper or not briefing:
        print("\n[-] Pipeline did not yield a valid executive briefing.")
        sys.exit(1)

    meta = briefing.get("metadata", {})
    print("\n" + "=" * 75)
    print("                   STRUCTURED EXECUTIVE BRIEFING")
    print("=" * 75)
    print(f"Title:        {meta.get('title')}")
    print(f"Authors:      {meta.get('authors')}")
    print(f"arXiv ID:     {meta.get('arxiv_id')}")
    print(f"Published:    {meta.get('published')}")
    print(f"PDF Link:     {meta.get('pdf_url')}")
    print("-" * 75)
    print(briefing.get("content_markdown"))
    print("=" * 75)

    # Stage 7: Interactive QA Loop
    print("\n[+] Entering Grounded QA Mode. Type your questions below (or 'exit' to quit):")
    if not state.get("retrieval_available", False):
        print("    [Notice: Answering solely from abstract/metadata due to PDF fallback]")

    while True:
        try:
            print()
            user_question = input("Ask a question about this paper: ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit", "q"]:
                print("\n[+] Exiting QA session. Goodbye!")
                break

            print("[*] Retrieving evidence chunks and generating grounded response...")
            state = ask_question_state(state, user_question)

            qa_history = state.get("qa_history", [])
            if not qa_history:
                print(f"[-] Could not generate answer: {state.get('error_message')}")
                continue

            last_turn = qa_history[-1]
            print("\n" + "-" * 50 + " ANSWER " + "-" * 50)
            print(last_turn.get("answer"))
            print("-" * 108)

            evidence = last_turn.get("evidence", [])
            if evidence:
                print(f"[Sources: {len(evidence)} evidence chunks retrieved from vector store]")

        except (KeyboardInterrupt, EOFError):
            print("\n\n[+] Session ended.")
            break

if __name__ == "__main__":
    main()
