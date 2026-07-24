"""Embedding Module — single source of truth for embedding generation.

Provides functions to generate text embeddings via Ollama and update
database rows with those embeddings.
"""

import logging
from typing import List, Optional

import psycopg2
import requests

from src.core.config import OLLAMA_URL, EMBED_MODEL, DB_CONFIG

logger = logging.getLogger(__name__)


def get_embedding(text: str) -> Optional[List[float]]:
    """Generate an embedding vector via Ollama.

    Args:
        text: Input text to embed (will be truncated to 4096 chars).

    Returns:
        A list of floats representing the embedding vector, or None on failure.

    Raises:
        requests.RequestException: If the Ollama API call fails.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except (requests.RequestException, KeyError, IndexError) as e:
        logger.error("Failed to get embedding: %s", e)
        raise


def update_embedding(row_id: int) -> None:
    """Fetch raw_text for a row, generate its embedding, and update the row.

    Args:
        row_id: The database row ID to update.

    Raises:
        psycopg2.Error: If the database operation fails.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT raw_text FROM invoices WHERE id = %s AND embedding IS NULL",
            (row_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            raw_text = row[0].strip()
            if raw_text:
                # Embed the full OCR text for richer semantic search.
                # Truncate to ~4K chars to stay within embedding model's context window.
                text_to_embed = raw_text[:4096]
                vec = get_embedding(text_to_embed)
                cur.execute(
                    "UPDATE invoices SET embedding = %s WHERE id = %s",
                    (vec, row_id),
                )
                conn.commit()
                logger.info("Embedding updated for row %d", row_id)
            else:
                logger.debug("Row %d has empty raw_text, skipping", row_id)
        else:
            logger.debug("Row %d already has embedding or does not exist", row_id)
    except Exception:
        conn.rollback()
        logger.exception("Failed to update embedding for row %d", row_id)
        raise
    finally:
        cur.close()
        conn.close()