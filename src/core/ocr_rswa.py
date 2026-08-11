"""R-SWA OCR Backend — Integration with llama-mtmd-cli from R-SWA branch.

This module provides OCR functionality using the R-SWA (Repetition/Avoidance) backend
via the llama-mtmd-cli command-line tool from the R-SWA branch (PR #24975).
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from src.core.config import (
    LLAMA_MTMD_CLI,
    UOCR_MODEL,
    UOCR_MMPROJ,
    UOCR_HF_REPO,
    RSWA_USE_HF,
    RSWA_GEN_PARAMS,
)

logger = logging.getLogger(__name__)


def run_ocr_rswa(image_path: str, prompt: str = "document parsing.", use_hf: bool = False) -> str:
    """Run OCR using the R-SWA backend via llama-mtmd-cli.

    Args:
        image_path: Path to the image file to process.
        prompt: OCR prompt to use (default: "document parsing.").
        use_hf: If True, use HuggingFace model instead of local model.

    Returns:
        Cleaned OCR text output.

    Raises:
        FileNotFoundError: If the llama-mtmd-cli binary is not found.
        subprocess.CalledProcessError: If the OCR command fails.
        subprocess.TimeoutExpired: If the OCR command times out.
    """
    # Validate input file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Validate CLI binary exists
    cli_path = Path(LLAMA_MTMD_CLI).expanduser()
    if not cli_path.exists():
        raise FileNotFoundError(
            f"llama-mtmd-cli binary not found at {LLAMA_MTMD_CLI}. "
            "Please build the R-SWA branch of llama.cpp or set LLAMA_MTMD_CLI environment variable."
        )

    # Build command line
    cmd = [
        str(cli_path),
        "--chat-template",
        "deepseek-ocr",
        "--image",
        image_path,
        "-p",
        prompt,
        "--temp",
        str(RSWA_GEN_PARAMS["temp"]),
        "--flash-attn",
        RSWA_GEN_PARAMS["flash_attn"],
        "-n",
        str(RSWA_GEN_PARAMS["n"]),
        "-c",
        str(RSWA_GEN_PARAMS["c"]),
        "--dry-multiplier",
        str(RSWA_GEN_PARAMS["dry_multiplier"]),
        "--dry-base",
        str(RSWA_GEN_PARAMS["dry_base"]),
        "--dry-allowed-length",
        str(RSWA_GEN_PARAMS["dry_allowed_length"]),
        "--dry-penalty-last-n",
        str(RSWA_GEN_PARAMS["dry_penalty_last_n"]),
        "--dry-sequence-breaker",
        RSWA_GEN_PARAMS["dry_sequence_breaker"],
    ]

    # Add model parameters
    if use_hf or RSWA_USE_HF:
        cmd.extend(["-hf", UOCR_HF_REPO])
        logger.info("Using HuggingFace model: %s", UOCR_HF_REPO)
    else:
        cmd.extend([
            "-m",
            UOCR_MODEL,
            "--mmproj",
            UOCR_MMPROJ,
        ])
        logger.debug("Using local model: %s with mmproj: %s", UOCR_MODEL, UOCR_MMPROJ)

    # Add optional flags
    if RSWA_GEN_PARAMS["no_warmup"]:
        cmd.append("--no-warmup")

    logger.debug("Running R-SWA OCR command: %s", " ".join(cmd))

    try:
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

        # Check for errors
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error("R-SWA OCR failed with return code %d: %s", result.returncode, error_msg)
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        # Process stdout to filter out logging lines
        stdout_lines = result.stdout.split("\n")
        filtered_lines = []

        for line in stdout_lines:
            line_stripped = line.strip()
            # Skip empty lines
            if not line_stripped:
                continue

            # Skip logging lines that start with:
            # - Digits (timestamps, numbers)
            # - "I " (info logging)
            # - "D " (debug logging)
            # - Common llama.cpp log prefixes
            if (
                line_stripped[0].isdigit()
                or line_stripped.startswith("I ")
                or line_stripped.startswith("D ")
                or any(line_stripped.startswith(prefix) for prefix in [
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
                ])
            ):
                continue

            filtered_lines.append(line_stripped)

        # Join filtered lines
        ocr_text = "\n".join(filtered_lines).strip()

        if not ocr_text:
            logger.warning("R-SWA OCR returned empty output for %s", image_path)
            return ""

        logger.info("R-SWA OCR completed for %s (%d chars)", image_path, len(ocr_text))
        return ocr_text

    except subprocess.TimeoutExpired as e:
        logger.error("R-SWA OCR timed out after 180 seconds for %s", image_path)
        raise
    except FileNotFoundError as e:
        logger.error("R-SWA OCR binary not found: %s", e)
        raise
    except Exception as e:
        logger.error("R-SWA OCR failed for %s: %s", image_path, e)
        raise