"""PDF Module — Convert PDF pages to images for OCR processing.

Provides functions to convert PDFs (from bytes or file path) into JPEG images
suitable for the OCR pipeline.
"""

import base64
import io
import logging
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image

from src.core.config import MAX_LONG_EDGE, JPEG_QUALITY

logger = logging.getLogger(__name__)

_TEMP_DIR = Path("/tmp")


def pdf_to_images_bytes(
    pdf_bytes: bytes, source_filename: str
) -> List[Tuple[str, str]]:
    """Convert a PDF (from bytes, e.g. Streamlit upload) to images.

    Args:
        pdf_bytes: Raw PDF file bytes.
        source_filename: Original filename for generating temp names.

    Returns:
        List of (temp_image_path, source_name) tuples.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_list: List[Tuple[str, str]] = []
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            w, h = img.size
            if max(w, h) > MAX_LONG_EDGE:
                ratio = MAX_LONG_EDGE / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            tmp = str(_TEMP_DIR / f"pdf_{source_filename}_page_{i}.jpg")
            img.save(tmp, "JPEG", quality=JPEG_QUALITY)
            page_list.append((tmp, f"{source_filename}_page_{i}"))
    finally:
        doc.close()
    logger.debug("Converted PDF %s to %d page images", source_filename, len(page_list))
    return page_list


def pdf_to_images_path(pdf_path: str, dpi: int = 200) -> List[str]:
    """Convert a PDF (from file path) to a list of temp image paths.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default: 200).

    Returns:
        List of temporary image file paths.
    """
    doc = fitz.open(pdf_path)
    page_paths: List[str] = []
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            w, h = img.size
            if max(w, h) > MAX_LONG_EDGE:
                ratio = MAX_LONG_EDGE / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            tmp = str(_TEMP_DIR / f"pdf_page_{Path(pdf_path).stem}_{i}.jpg")
            img.save(tmp, "JPEG", quality=JPEG_QUALITY)
            page_paths.append(tmp)
    finally:
        doc.close()
    logger.debug("Converted PDF %s to %d page images", pdf_path, len(page_paths))
    return page_paths


def pdf_pages_to_base64_list(pdf_path: str) -> List[str]:
    """Convert each page of a PDF to a base64-encoded JPEG string.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of base64-encoded JPEG strings.
    """
    b64_list: List[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            w, h = img.size
            if max(w, h) > MAX_LONG_EDGE:
                ratio = MAX_LONG_EDGE / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            b64_list.append(base64.b64encode(buf.getvalue()).decode())
    finally:
        doc.close()
    return b64_list