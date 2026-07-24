"""Application configuration loaded from environment variables.

All configuration values are read from environment variables (with sensible defaults).
Loads a .env file from the project root if present.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────
# Walk up from this file's location to find the project root
_core_dir = Path(__file__).resolve().parent
_src_dir = _core_dir.parent
_project_root = _src_dir.parent
_dotenv_path = _project_root / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)
else:
    # Fallback: try loading from current working directory
    load_dotenv()

# ── Logging setup ────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.debug("Configuration loaded from %s", _dotenv_path if _dotenv_path.exists() else "environment variables")

# ── OCR: Server mode (primary, faster) ───────────────────────────────────
OCR_MODE: str = os.getenv("OCR_MODE", "server")
OCR_SERVER_URL: str = os.getenv("OCR_SERVER_URL", "http://127.0.0.1:8081/v1/chat/completions")
OCR_SERVER_MODEL: str = os.getenv("OCR_SERVER_MODEL", "Unlimited-OCR")
OCR_SERVER_PROMPT: str = os.getenv("OCR_SERVER_PROMPT", "Please OCR the text in this image.")
OCR_SERVER_TEMPERATURE: float = float(os.getenv("OCR_SERVER_TEMPERATURE", "0.1"))
OCR_SERVER_MAX_TOKENS: int = int(os.getenv("OCR_SERVER_MAX_TOKENS", "2048"))
OCR_SERVER_REPEAT_PENALTY: float = float(os.getenv("OCR_SERVER_REPEAT_PENALTY", "1.5"))

# ── OCR: CLI mode (fallback, uses subprocess) ────────────────────────────
LLAMA_CLI: str = os.getenv("LLAMA_CLI", os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli"))
UOCR_MODEL: str = os.getenv("UOCR_MODEL", os.path.expanduser("~/uocr/Unlimited-OCR-Q4_K_M.gguf"))
UOCR_MMPROJ: str = os.getenv("UOCR_MMPROJ", os.path.expanduser("~/uocr/mmproj-Unlimited-OCR-F16.gguf"))

# ── Image preprocessing ──────────────────────────────────────────────────
MAX_LONG_EDGE: int = int(os.getenv("MAX_LONG_EDGE", "1024"))
JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "85"))

# ── Ollama ───────────────────────────────────────────────────────────────
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TEXT_MODEL: str = os.getenv("TEXT_MODEL", "qwen2.5:1.5b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "mxbai-embed-large")
RAG_MODEL: str = os.getenv("RAG_MODEL", "qwen2.5:1.5b")

# ── llama-server for multi-page mode ─────────────────────────────────────
LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8081/v1/chat/completions")

# ── PostgreSQL ───────────────────────────────────────────────────────────
DB_CONFIG: Dict[str, Any] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "ocr"),
    "password": os.getenv("DB_PASSWORD", "***REMOVED***"),
    "dbname": os.getenv("DB_NAME", "invoices"),
}

# ── JSON extraction prompts ──────────────────────────────────────────────
JSON_PROMPT: str = """Return a single JSON object with these keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".

Rules:
- Use the exact text from the invoice. Do NOT invent or guess any values.
- If a field is missing, set it to "".
- "total_amount" must contain only the number (e.g. "1250.00"), without currency symbol.
- "currency" must be the three‑letter currency code (e.g. "USD").
- Do NOT use nested objects.

Invoice text:
___RAW_TEXT___

JSON:"""

FALLBACK_PROMPT: str = """Extract these fields from the invoice text.
Do NOT use any of the following words: value, text, string, example, placeholder, xxxx.
Return ONLY a valid JSON object with the keys:
"invoice_number", "date", "vendor_name", "total_amount", "currency".
"total_amount" must be a plain number (e.g. "1250.00").
"currency" must be a three‑letter code (e.g. "USD").
If a field is truly missing, leave it as "".

Invoice text:
___RAW_TEXT___

JSON:"""