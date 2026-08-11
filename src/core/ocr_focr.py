"""FOCR (Franken OCR) Backend — CPU-optimized OCR via focr CLI.

This module provides OCR functionality using the focr Rust CLI tool,
which natively supports PDF multipage processing via --multi-page.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Union

from src.core.config import (
    FOCR_EXECUTABLE,
    FOCR_MODEL,
    FOCR_TEMPERATURE,
    FOCR_MAX_LENGTH,
    FOCR_NO_REPEAT_NGRAM,
    FOCR_NGRAM_WINDOW,
    FOCR_CROP_MODE,
    FOCR_BASE_SIZE,
    FOCR_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


def run_ocr_focr(image_path: str, multi_page: bool = False) -> str:
    """Run OCR using the focr CLI tool.

    Shows real-time progress from focr's stderr output.

    Args:
        image_path: Path to image or PDF file.
        multi_page: If True, use --multi-page for cross-page processing (PDFs only).

    Returns:
        OCR text output (markdown format).

    Raises:
        FileNotFoundError: If focr binary or input file not found.
        subprocess.CalledProcessError: If focr command fails.
        subprocess.TimeoutExpired: If focr times out.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input file not found: {image_path}")

    # Check if focr is available
    try:
        subprocess.run([FOCR_EXECUTABLE, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise FileNotFoundError(
            f"focr binary not found at '{FOCR_EXECUTABLE}'. "
            "Install from: https://github.com/franken-ocr/franken_ocr"
        )

    is_pdf = image_path.lower().endswith(".pdf")
    cmd = [FOCR_EXECUTABLE, "ocr"]

    # Model
    if FOCR_MODEL:
        cmd.extend(["--model", FOCR_MODEL])

    # Generation parameters
    if FOCR_TEMPERATURE > 0:
        cmd.extend(["--temperature", str(FOCR_TEMPERATURE)])
    cmd.extend(["--max-length", str(FOCR_MAX_LENGTH)])
    cmd.extend(["--no-repeat-ngram", str(FOCR_NO_REPEAT_NGRAM)])
    cmd.extend(["--ngram-window", str(FOCR_NGRAM_WINDOW)])
    cmd.extend(["--crop-mode", FOCR_CROP_MODE])
    cmd.extend(["--base-size", str(FOCR_BASE_SIZE)])
    cmd.extend(["--image-size", str(FOCR_IMAGE_SIZE)])

    # Multipage handling
    if multi_page and is_pdf:
        cmd.append("--multi-page")

    # Input file
    cmd.append(image_path)

    logger.info("Running FOCR on %s (multi_page=%s)", image_path, multi_page)
    logger.debug("FOCR command: %s", " ".join(cmd))

    try:
        # Run with real-time stderr streaming for progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_lines = []
        stderr_lines = []

        # Read stderr in real-time to show progress
        if process.stderr:
            for line in process.stderr:
                line = line.strip()
                if line:
                    stderr_lines.append(line)
                    # Show progress lines that contain useful info
                    if any(keyword in line.lower() for keyword in [
                        "page", "loading", "processing", "ocr", "model",
                        "progress", "percent", "%", "error", "warning"
                    ]):
                        print(f"  [FOCR] {line}", flush=True)

        # Wait for completion
        stdout, _ = process.communicate(timeout=600)
        returncode = process.returncode

        if returncode != 0:
            error_msg = "\n".join(stderr_lines) if stderr_lines else "Unknown error"
            logger.error("FOCR failed with return code %d: %s", returncode, error_msg)
            print(f"  ❌ FOCR error (exit {returncode}): {error_msg}", flush=True)
            raise subprocess.CalledProcessError(
                returncode,
                cmd,
                output=stdout,
                stderr="\n".join(stderr_lines),
            )

        stdout = stdout.strip()
        if not stdout:
            logger.warning("FOCR returned empty output for %s", image_path)
            return ""

        logger.info("FOCR completed for %s (%d chars)", image_path, len(stdout))
        return stdout

    except subprocess.TimeoutExpired:
        logger.error("FOCR timed out after 600 seconds for %s", image_path)
        raise
    except Exception as e:
        logger.error("FOCR failed for %s: %s", image_path, e)
        raise


def run_ocr_focr_batch(image_paths: List[str], multi_page: bool = False) -> str:
    """Run FOCR batch mode on multiple images.

    Args:
        image_paths: List of image file paths.
        multi_page: If True, treat images as one multi-page document.

    Returns:
        Combined OCR output (markdown format).
    """
    if not image_paths:
        return ""

    # Check if focr is available
    try:
        subprocess.run([FOCR_EXECUTABLE, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise FileNotFoundError(
            f"focr binary not found at '{FOCR_EXECUTABLE}'. "
            "Install from: https://github.com/franken-ocr/franken_ocr"
        )

    cmd = [FOCR_EXECUTABLE, "ocr-batch"]

    # Model
    if FOCR_MODEL:
        cmd.extend(["--model", FOCR_MODEL])

    # Generation parameters
    if FOCR_TEMPERATURE > 0:
        cmd.extend(["--temperature", str(FOCR_TEMPERATURE)])
    cmd.extend(["--max-length", str(FOCR_MAX_LENGTH)])
    cmd.extend(["--no-repeat-ngram", str(FOCR_NO_REPEAT_NGRAM)])
    cmd.extend(["--ngram-window", str(FOCR_NGRAM_WINDOW)])
    cmd.extend(["--crop-mode", FOCR_CROP_MODE])
    cmd.extend(["--base-size", str(FOCR_BASE_SIZE)])
    cmd.extend(["--image-size", str(FOCR_IMAGE_SIZE)])

    # Multipage handling
    if multi_page:
        cmd.append("--multi-page")

    # Add all image paths
    cmd.extend(image_paths)

    logger.info("Running FOCR batch on %d images (multi_page=%s)", len(image_paths), multi_page)
    logger.debug("FOCR batch command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error("FOCR batch failed: %s", error_msg)
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        stdout = result.stdout.strip()
        if not stdout:
            logger.warning("FOCR batch returned empty output")
            return ""

        logger.info("FOCR batch completed (%d chars)", len(stdout))
        return stdout

    except subprocess.TimeoutExpired:
        logger.error("FOCR batch timed out after 600 seconds")
        raise
    except Exception as e:
        logger.error("FOCR batch failed: %s", e)
        raise


def extract_text_from_focr_output(output: str) -> str:
    """Extract clean text from focr output by removing image references.

    FOCR outputs markdown with image references like ![](images/0.jpg).
    For invoice OCR we want just the text content.

    Args:
        output: Raw focr output (markdown or JSON).

    Returns:
        Cleaned text without image references.
    """
    # If JSON, extract markdown field
    if output.strip().startswith("{"):
        try:
            data = json.loads(output)
            markdown = data.get("markdown", output)
        except json.JSONDecodeError:
            markdown = output
    else:
        markdown = output

    # Remove image references like ![](images/...)
    import re
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', markdown)
    # Remove empty lines resulting from image removal
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    # Remove <PAGE> separators
    cleaned = cleaned.replace('<PAGE>', '\n')

    return cleaned.strip()