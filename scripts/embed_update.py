#!/usr/bin/env python3
"""Batch embedding updater — generate embeddings for all rows missing them.

Usage:
  python scripts/embed_update.py
"""

import logging
import sys
from pathlib import Path

# Ensure the package is importable when running from the scripts directory
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import psycopg2

from src.core.config import DB_CONFIG
from src.core.embedding import get_embedding

logger = logging.getLogger(__name__)


def main() -> None:
    """Fetch all rows without embeddings and generate them."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Find rows missing embeddings
    cur.execute(
        "SELECT id, raw_text FROM invoices WHERE embedding IS NULL AND raw_text IS NOT NULL"
    )
    rows = cur.fetchall()
    total = len(rows)
    print(f"📊 Found {total} row(s) without embeddings.")

    for i, (row_id, raw_text) in enumerate(rows, 1):
        text = (raw_text or "").strip()
        if not text:
            print(f"  ⏭️  [{i}/{total}] Row {row_id}: empty raw_text, skipping.")
            continue

        try:
            # Truncate to ~4K chars to stay within embedding model's context window
            text_to_embed = text[:4096]
            vec = get_embedding(text_to_embed)
            cur.execute("UPDATE invoices SET embedding = %s WHERE id = %s", (vec, row_id))
            conn.commit()
            print(f"  ✅ [{i}/{total}] Row {row_id}: embedding generated.")
        except Exception as e:
            conn.rollback()
            print(f"  ❌ [{i}/{total}] Row {row_id}: failed - {e}")

    cur.close()
    conn.close()
    print(f"\n🎉 Embedding update complete ({total} rows processed).")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()