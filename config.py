"""All settings in one place. Values can be overridden from the .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).parent


def _load_env():
    """Tiny .env reader (so we don't need an extra package)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

# ---- Folders / files -------------------------------------------------------
DOCS_DIR = ROOT / "docs"            # the 4 knowledge-base documents
DATA_DIR = ROOT / "data"            # generated: text versions, chunks, embeddings
TEXT_DIR = DATA_DIR / "text"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMB_FILE = DATA_DIR / "embeddings.npy"
LOG_DIR = ROOT / "logs"
QUERY_LOG = LOG_DIR / "queries.jsonl"

# ---- Chunking / retrieval --------------------------------------------------
MAX_WORDS_PER_CHUNK = 180
TOP_K = int(os.getenv("TOP_K", "4"))
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "1") == "1"
USE_OCR = os.getenv("USE_OCR", "1") == "1"            # read text inside screenshots (ingest only)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")        # only if Tesseract is in an unusual folder
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ---- LLM (all three providers speak the same "OpenAI chat" protocol) --------
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:3b",
        "key_env": None,
    },
}

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
if PROVIDER not in PROVIDERS:
    raise SystemExit(f"LLM_PROVIDER must be one of {list(PROVIDERS)} (got '{PROVIDER}')")

_preset = PROVIDERS[PROVIDER]
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or _preset["base_url"]
LLM_MODEL = os.getenv("LLM_MODEL") or _preset["model"]
KEY_ENV_NAME = _preset["key_env"]
LLM_API_KEY = os.getenv("LLM_API_KEY") or (os.getenv(KEY_ENV_NAME, "") if KEY_ENV_NAME else "")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "350"))

PORT = int(os.getenv("PORT", "8000"))
