"""Extraction Module — JSON extraction from OCR text using Ollama LLM.

Provides functions to extract structured invoice data from raw OCR text,
with fallback prompts and validation.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

import requests

from src.core.config import OLLAMA_URL, TEXT_MODEL, JSON_PROMPT, FALLBACK_PROMPT

logger = logging.getLogger(__name__)


def _normalise_date(date_str: str) -> str:
    """Normalise various date formats to YYYY-MM-DD.

    Args:
        date_str: A date string in various formats.

    Returns:
        Normalised date string in YYYY-MM-DD format, or the original if unrecognised.
    """
    if not date_str:
        return ""
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{b:02d}-{a:02d}" if a > 12 else f"{y:04d}-{a:02d}-{b:02d}"
    m = re.match(r'^(\d{1,2})[- ]([A-Za-z]{3})[- ](\d{4})$', date_str)
    if m:
        months = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        mm = months.get(m.group(2).lower())
        if mm:
            return f"{int(m.group(3)):04d}-{mm}-{int(m.group(1)):02d}"
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', date_str)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', date_str)
    if m:
        return f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return date_str


def clean_invoice_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Force all fields into the correct format.

    Args:
        data: Raw extracted invoice data dictionary.

    Returns:
        Cleaned dictionary with standardised field formats.
    """
    ta = data.get("total_amount")
    if isinstance(ta, dict):
        amount = ta.get("amount", "")
        curr = ta.get("currency", "")
        data["total_amount"] = f"{float(amount):.2f}" if amount else ""
        if curr and not data.get("currency"):
            data["currency"] = curr
    elif isinstance(ta, str):
        cleaned = re.sub(r'[^\d.]', '', ta.replace(',', '').replace(' ', ''))
        data["total_amount"] = f"{float(cleaned):.2f}" if cleaned else ""
    elif isinstance(ta, (int, float)):
        data["total_amount"] = f"{float(ta):.2f}"
    else:
        data["total_amount"] = ""

    curr = data.get("currency", "")
    if isinstance(curr, str):
        curr = curr.strip().upper()
        match = re.match(r'^([A-Z]{3})', curr)
        data["currency"] = match.group(1) if match else ""
    else:
        data["currency"] = ""

    date_val = data.get("date", "")
    if date_val:
        data["date"] = _normalise_date(str(date_val).strip())
    else:
        data["date"] = ""

    for key in ["invoice_number", "date", "vendor_name", "total_amount", "currency"]:
        if key not in data:
            data[key] = ""
    return data


def _extract_json(prompt_template: str, raw_text: str) -> Optional[Dict[str, Any]]:
    """Send a prompt to Ollama and parse the JSON response.

    Args:
        prompt_template: The prompt template with ___RAW_TEXT___ placeholder.
        raw_text: The OCR text to extract from.

    Returns:
        Parsed and cleaned invoice data dict, or None if parsing fails.
    """
    prompt = prompt_template.replace("___RAW_TEXT___", raw_text)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "")
        start = content.index('{')
        end = content.rindex('}') + 1
        data = json.loads(content[start:end])
        data = clean_invoice_data(data)
        logger.debug("Extracted JSON from OCR text (%d chars)", len(raw_text))
        return data
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse JSON from LLM response: %s", e)
        return None
    except requests.RequestException as e:
        logger.error("Ollama request failed during extraction: %s", e)
        return None


def text_to_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract structured invoice data from OCR text using the primary prompt.

    Args:
        raw_text: The raw OCR text.

    Returns:
        Cleaned invoice data dict, or None if extraction fails.
    """
    return _extract_json(JSON_PROMPT, raw_text)


def text_to_json_fallback(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract structured invoice data using a stricter fallback prompt.

    Args:
        raw_text: The raw OCR text.

    Returns:
        Cleaned invoice data dict, or None if extraction fails.
    """
    return _extract_json(FALLBACK_PROMPT, raw_text)


def is_likely_fake(data: Optional[Dict[str, Any]]) -> bool:
    """Check if extracted data contains placeholder/fake values.

    Args:
        data: The extracted invoice data dict.

    Returns:
        True if any field contains a suspicious placeholder value.
    """
    if data is None:
        return True
    suspicious = {"value", "text", "string", "example", "placeholder", "xxxx"}
    for field in ["invoice_number", "date", "vendor_name", "total_amount"]:
        val = (data.get(field) or "").strip().lower()
        if val in suspicious:
            return True
    return False