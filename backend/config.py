import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage configuration
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", DATA_DIR / "chroma"))
PDF_STORAGE_DIR = Path(os.getenv("PDF_STORAGE_DIR", DATA_DIR / "papers"))

# Ensure directories exist
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# LLM Provider Hierarchy: Groq (Primary) -> Gemini (Fallback) -> Mock (Offline)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
FALLBACK_LLM_PROVIDER = os.getenv("FALLBACK_LLM_PROVIDER", "gemini").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Local Embedding Model (Runs locally on CPU)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval & Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
