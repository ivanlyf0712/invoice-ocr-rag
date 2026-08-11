#!/usr/bin/env python3
"""Fast two‑stage invoice OCR pipeline with PDF support.

Usage:
  python -m src.invoice.pipeline -f invoice.jpg
  python -m src.invoice.pipeline -f batch_of_invoices.pdf
  python -m src.invoice.pipeline -d ./input_folder/
  python -m src.invoice.pipeline -f invoice.jpg --ocr-only    # Print OCR text only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import psycopg2
import requests

from src.core.config import OLLAMA_URL, DB_CONFIG, OCR_BACKEND
from src.core.ocr import run_ocr, clean_grounding_tags
from src.core.ocr_focr import run_ocr_focr, extract_text_from_focr_output
from src.core.extraction import text_to_json, is_likely_fake
from src.core.embedding import get_embedding
from src.core.pdf import pdf_to_images_path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATABASE INSERT (with inline embedding)
# ═══════════════════════════════════════════════════════════════════

def insert_into_db(fields: Dict[str, Any], raw_text: str, source_file: str) -> None:
    """Insert an invoice record and generate its embedding.

    Args:
        fields: Dictionary with invoice fields.
        raw_text: The raw OCR text.
        source_file: The source filename.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO invoices (invoice_number, date, vendor_name,
                                     total_amount, currency, raw_text, source_file)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                fields.get("invoice_number", ""),
                fields.get("date", ""),
                fields.get("vendor_name", ""),
                fields.get("total_amount", ""),
                fields.get("currency", ""),
                raw_text,
                source_file,
            ),
        )
        new_id = cur.fetchone()[0]
        if raw_text and raw_text.strip():
            vec = get_embedding(raw_text.strip())
            cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, new_id))
        conn.commit()
        print(f"  ✅ Inserted: {source_file} (id={new_id}, embedding generated)")
    except Exception as e:
        conn.rollback()
        print(f"  ❌ DB error: {e}")
    finally:
        cur.close()
        conn.close()


# ═══════════════════════════════════════════════════════════════════
# PROCESS SINGLE IMAGE (OR PDF PAGE)
# ═══════════════════════════════════════════════════════════════════

def process_single_image(image_path: str, source_file: Optional[str] = None) -> Optional[str]:
    """Process a single image: OCR, extract JSON, and optionally insert into DB.

    Args:
        image_path: Path to the image file.
        source_file: Source filename (defaults to basename of image_path).

    Returns:
        The raw OCR text if --ocr-only mode, otherwise None.
    """
    if source_file is None:
        source_file = os.path.basename(image_path)

    fname = source_file
    print(f"\n📄 {fname}")
    t0 = time.time()

    raw_text = run_ocr(image_path)
    raw_text = clean_grounding_tags(raw_text)
    print("==============OCR Result===================")
    print(raw_text)
    t1 = time.time()
    print(f"  ⏱  OCR: {t1-t0:.1f}s")
    print("===========================================")

    # If --ocr-only, just return the text without DB insertion
    if _OCR_ONLY_MODE:
        return raw_text

    data = text_to_json(raw_text)
    t2 = time.time()
    print(f"  ⏱  JSON parse: {t2-t1:.1f}s")

    if data is None:
        print("  ⚠️  JSON extraction failed – inserting raw text only.")
        data = {}
    else:
        print(f"  📊 Fields: {json.dumps(data, indent=2)}")

    empty_count = 0
    for key in ("invoice_number", "date", "vendor_name", "currency"):
        if not data.get(key):
            empty_count += 1
    ta = data.get("total_amount")
    if ta is None or ta == 0 or ta == 0.0 or ta == "":
        empty_count += 1
    if empty_count > 2:
        print(f"  🚫 Rejected: {empty_count}/5 fields empty – inserting with fields cleared.")
        data = {"invoice_number": "", "date": "", "vendor_name": "", "total_amount": 0.0, "currency": ""}

    insert_into_db(data, raw_text, fname)
    print(f"  🕐 Total: {time.time()-t0:.1f}s")
    return None


# Global flag set by argparse
_OCR_ONLY_MODE = False


# ═══════════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ═══════════════════════════════════════════════════════════════════

def process_image(image_path: str) -> None:
    """Process a single image or PDF file.

    Args:
        image_path: Path to the image or PDF file.
    """
    fname = os.path.basename(image_path)

    # Check if FOCR backend is enabled
    use_focr = OCR_BACKEND.lower() == "focr"

    if image_path.lower().endswith(".pdf"):
        if use_focr:
            # ── FOCR multipage PDF processing ──────────────────────────
            print(f"\n📑 PDF detected: {fname} (FOCR multipage mode)")
            t0 = time.time()
            try:
                raw_text = run_ocr_focr(image_path, multi_page=True)
                raw_text = extract_text_from_focr_output(raw_text)
                t1 = time.time()
                print(f"  ⏱  OCR: {t1-t0:.1f}s")
                print("==============OCR Result===================")
                print(raw_text)
                print("===========================================")

                # If --ocr-only, just return the text without DB insertion
                if _OCR_ONLY_MODE:
                    return

                data = text_to_json(raw_text)
                t2 = time.time()
                print(f"  ⏱  JSON parse: {t2-t1:.1f}s")

                if data is None:
                    print("  ⚠️  JSON extraction failed – inserting raw text only.")
                    data = {}
                else:
                    print(f"  📊 Fields: {json.dumps(data, indent=2)}")

                insert_into_db(data, raw_text, fname)
                print(f"  🕐 Total: {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  ❌ FOCR failed: {e}")
                raise
        else:
            # ── Legacy page-by-page processing ──────────────────────────
            print(f"\n📑 PDF detected: {fname} (page‑by‑page mode)")
            page_paths = pdf_to_images_path(image_path)
            total_pages = len(page_paths)
            print(f"   → {total_pages} pages.")
            for i, page_path in enumerate(page_paths):
                source = f"{fname}_page_{i}"
                try:
                    process_single_image(page_path, source)
                except Exception as e:
                    print(f"  ❌ Page {i} failed: {e}")
                os.remove(page_path)
                print(f"  [{i+1}/{total_pages}] pages done.")
            print(f"  🎉 PDF processing complete ({total_pages} pages).")
        return

    # Single image processing
    if use_focr:
        # FOCR for single images
        print(f"\n📄 {fname} (FOCR mode)")
        t0 = time.time()
        try:
            raw_text = run_ocr_focr(image_path, multi_page=False)
            raw_text = extract_text_from_focr_output(raw_text)
            t1 = time.time()
            print(f"  ⏱  OCR: {t1-t0:.1f}s")
            print("==============OCR Result===================")
            print(raw_text)
            print("===========================================")

            if _OCR_ONLY_MODE:
                return

            data = text_to_json(raw_text)
            t2 = time.time()
            print(f"  ⏱  JSON parse: {t2-t1:.1f}s")

            if data is None:
                print("  ⚠️  JSON extraction failed – inserting raw text only.")
                data = {}
            else:
                print(f"  📊 Fields: {json.dumps(data, indent=2)}")

            insert_into_db(data, raw_text, fname)
            print(f"  🕐 Total: {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  ❌ FOCR failed: {e}")
            raise
    else:
        # Legacy single image processing
        process_single_image(image_path)


def main():
    """Main entry point for the OCR pipeline CLI."""
    global _OCR_ONLY_MODE

    parser = argparse.ArgumentParser(
        description="Fast two‑stage OCR pipeline (Unlimited‑OCR + Qwen2.5‑1.5B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f invoice.jpg
  %(prog)s -f batch.pdf
  %(prog)s -d ~/invoices/
  %(prog)s -f invoice.jpg --ocr-only
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Single invoice image or PDF")
    group.add_argument("-d", "--dir", help="Directory containing invoice images/PDFs")
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Only run OCR and print text, do not insert into database.",
    )
    parser.add_argument(
        "--backend",
        choices=["llama", "focr", "rswa"],
        default=None,
        help="OCR backend to use: 'llama' (default), 'focr' (Franken OCR), or 'rswa' (R-SWA). "
             "Overrides OCR_BACKEND environment variable.",
    )
    args = parser.parse_args()

    # Override OCR_BACKEND if --backend is specified
    if args.backend:
        os.environ["OCR_BACKEND"] = args.backend

    _OCR_ONLY_MODE = args.ocr_only

    if args.file:
        try:
            if args.ocr_only:
                text = process_single_image(args.file)
                if text:
                    print("\n" + "=" * 60)
                    print("OCR TEXT OUTPUT (--ocr-only mode)")
                    print("=" * 60)
                    print(text)
                    print("=" * 60)
            else:
                process_image(args.file)
        except Exception as e:
            print(f"  ❌ Failed to process {args.file}: {e}")
            sys.exit(1)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}
        skipped = []
        for root, _, files in os.walk(args.dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in exts:
                    filepath = os.path.join(root, fname)
                    try:
                        process_image(filepath)
                    except Exception as e:
                        print(f"  ❌ Skipped {filepath}: {e}")
                        skipped.append(filepath)
        if skipped:
            print(f"\n⚠️  Skipped {len(skipped)} file(s) due to errors:")
            for s in skipped:
                print(f"    - {s}")


if __name__ == "__main__":
    main()