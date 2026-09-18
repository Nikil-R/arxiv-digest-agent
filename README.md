# Autonomous arXiv Paper Digest & QA Agent

An autonomous, stateful agent that processes research topics or arXiv paper IDs, downloads and parses PDFs, produces structured executive briefings, and enables interactive grounded Q&A backed by a local vector store.

## Overview
- **Agentic State Graph**: Explicit stateful pipeline (nodes, edges, shared state) with deterministic control flow.
- **Retrieval & Parsing**: Official arXiv Atom API integration, section-aware PDF text extraction with PyMuPDF.
- **Local Vector Search**: Chunking and embedding stored locally via ChromaDB.
- **Grounded Q&A**: Strict citation-backed responses preventing hallucination.
- **Zero-Cost & Offline Ready**: Runs with free-tier LLM providers or in an offline deterministic mode with no paid API keys required.

## Status
*Under active milestone-based development.*
