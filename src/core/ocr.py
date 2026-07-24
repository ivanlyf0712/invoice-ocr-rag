"""OCR Module — OCR engine supporting two backends.

Backends:
  - Server mode (default): sends images to llama-server via HTTP
  - CLI mode (fallback): calls llama-mtmd-cli via subprocess

Includes retry logic with exponential backoff for transient server failures.
"""

import base64
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from src.core.config import (
    OCR_MODE,
    OCR_SERVER_URL,
    OCR_SERVER_MODEL,
    OCR_SERVER_PROMPT,
    OCR_SERVER_TEMPERATURE,
    OCR_SERVER_MAX_TOKENS,
    OCR_SERVER_REPEAT_PENALTY,
    LLAMA_CLI,
    UOCR_MODEL,
    UOCR_MMPROJ,
    MAX_LONG_EDGE,
    JPEG_QUALITY,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_TEMP_DIR = Path("/tmp")


# ── Image preprocessing ──────────────────────────────────────────────────

def _preprocess_image(image_path: str) -> str:
    """Resize and compress an image, return path to temporary JPEG.

    Args:
        image_path: Path to the source image.

    Returns:
        Path to the preprocessed temporary JPEG file.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_EDGE:
        ratio = MAX_LONG_EDGE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    tmp = str(_TEMP_DIR / "ocr_server.jpg")
    img.save(tmp, "JPEG", quality=JPEG_QUALITY)
    logger.debug("Preprocessed image %s -> %s (size: %dx%d)", image_path, tmp, img.width, img.height)
    return tmp


# ── Text cleaning ────────────────────────────────────────────────────────

def clean_grounding_tags(text: str) -> str:
    """Remove grounding markers left by CLI-mode OCR.

    Args:
        text: Raw OCR text with potential grounding markers.

    Returns:
        Cleaned text without grounding markers.
    """
    cleaned = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    cleaned = re.sub(r'<\|(?:grounding|ref|det|/det|/ref)\|>', '', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    return cleaned.strip()


def _clean_server_text(text: str) -> str:
    """Remove coordinate annotations from server-mode OCR output.

    Server returns lines like: "title [59, 228, 224, 263]INVOICE"
    We want just: "INVOICE"

    Args:
        text: Raw server-mode OCR output.

    Returns:
        Cleaned text without coordinate annotations.
    """
    cleaned = re.sub(r'\b(?:title|text|para|line|block)\s*\[[\d,\s]+\]\s*', '', text)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    return cleaned.strip()


# ── Server-mode OCR (primary) ────────────────────────────────────────────

def run_ocr_server(image_path: str) -> str:
    """Send an image to llama-server, return cleaned OCR text.

    Implements retry logic with exponential backoff for transient failures.

    Args:
        image_path: Path to the image file.

    Returns:
        Cleaned OCR text.

    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    tmp = _preprocess_image(image_path)
    try:
        with open(tmp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    finally:
        os.unlink(tmp)

    payload = {
        "model": OCR_SERVER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_SERVER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "temperature": OCR_SERVER_TEMPERATURE,
        "max_tokens": OCR_SERVER_MAX_TOKENS,
        "repeat_penalty": OCR_SERVER_REPEAT_PENALTY,
        "stream": False,
        "cache_prompt": False,
    }

    last_exception: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.debug("OCR server attempt %d/%d for %s", attempt, _MAX_RETRIES, image_path)
            resp = requests.post(OCR_SERVER_URL, json=payload, timeout=180)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.info("OCR server succeeded on attempt %d/%d", attempt, _MAX_RETRIES)
            return content.strip()
        except (requests.RequestException, KeyError, IndexError) as e:
            last_exception = e
            logger.warning(
                "OCR server attempt %d/%d failed: %s. %s",
                attempt,
                _MAX_RETRIES,
                e,
                "Retrying..." if attempt < _MAX_RETRIES else "No more retries.",
            )
            if attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)

    raise last_exception  # type: ignore[misc]


# ── CLI-mode OCR (fallback) ──────────────────────────────────────────────

def run_ocr_cli(image_path: str) -> str:
    """Run Unlimited-OCR via subprocess (legacy, kept for environments
    where llama-server is not running).

    Args:
        image_path: Path to the image file.

    Returns:
        Cleaned OCR text from CLI output.
    """
    processed = _preprocess_image(image_path)
    cmd = [
        LLAMA_CLI,
        "-m",
        UOCR_MODEL,
        "--mmproj",
        UOCR_MMPROJ,
        "--image",
        processed,
        "-p",
        "Free OCR.",
        "--chat-template",
        "deepseek-ocr",
        "--temp",
        "0",
        "-c",
        "2048",
        "-ngl",
        "0",
        "--threads",
        "4",
        "-n",
        "384",
    ]
    logger.debug("Running CLI OCR: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    stdout = result.stdout.decode("utf-8", errors="replace")
    skip = [
        "llama_model_loader",
        "llama_model_load",
        "encode_image",
        "system_info",
        "main:",
        "init:",
        "build:",
        "start:",
        "clip_model",
        "ggml_",
        "warming up",
        "srv",
        "slot",
        "kv_cache",
    ]
    lines = [l.strip() for l in stdout.split("\n") if l.strip() and not any(s in l for s in skip)]
    text = "\n".join(lines)
    logger.info("CLI OCR completed for %s (%d chars)", image_path, len(text))
    return text


# ── Public API ───────────────────────────────────────────────────────────

def run_ocr(image_path: str) -> str:
    """OCR an image using the active backend (configured via OCR_MODE).

    Args:
        image_path: Path to the image file.

    Returns:
        Cleaned OCR text.

    Raises:
        RuntimeError: If OCR fails in both modes.
    """
    logger.info("Running OCR on %s (mode=%s)", image_path, OCR_MODE)
    if OCR_MODE == "cli":
        text = run_ocr_cli(image_path)
        return clean_grounding_tags(text)
    else:
        return _clean_server_text(run_ocr_server(image_path))