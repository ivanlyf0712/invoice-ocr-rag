"""Database Module — PostgreSQL operations for invoice storage and retrieval.

Uses psycopg2 connection pooling for improved performance.
Embedding functions are imported from src.core.embedding (single source of truth).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import requests
import warnings

from src.core.config import DB_CONFIG, OLLAMA_URL, EMBED_MODEL
from src.core.embedding import get_embedding, update_embedding

logger = logging.getLogger(__name__)

# Suppress pandas+psycopg2 warning (pd.read_sql works fine with raw connections)
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

# ── Connection Pool ──────────────────────────────────────────────────────
_pool: Optional[SimpleConnectionPool] = None


def _get_pool() -> SimpleConnectionPool:
    """Get or create the database connection pool (singleton)."""
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)
        logger.info("Database connection pool created (min=1, max=10)")
    return _pool


def get_db_connection():
    """Get a database connection from the pool.

    Returns:
        A psycopg2 connection from the pool.

    Raises:
        psycopg2.Error: If no connection is available.
    """
    pool = _get_pool()
    conn = pool.getconn()
    logger.debug("Acquired database connection from pool")
    return conn


def put_db_connection(conn) -> None:
    """Return a connection to the pool.

    Args:
        conn: The psycopg2 connection to return.
    """
    if _pool is not None:
        _pool.putconn(conn)
        logger.debug("Returned database connection to pool")


def close_all_connections() -> None:
    """Close all connections in the pool (call on application shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        logger.info("All database connections closed")
        _pool = None


# ── Invoice CRUD ─────────────────────────────────────────────────────────

def insert_invoice(fields: Dict[str, Any], raw_text: str, source_file: str) -> int:
    """Insert a new invoice record into the database.

    Args:
        fields: Dictionary with keys invoice_number, date, vendor_name,
                total_amount, currency.
        raw_text: The raw OCR text.
        source_file: The source filename.

    Returns:
        The new row ID.

    Raises:
        psycopg2.Error: If the insert fails.
    """
    conn = get_db_connection()
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
        conn.commit()
        logger.info("Inserted invoice id=%d from %s", new_id, source_file)
        return new_id
    except Exception as e:
        conn.rollback()
        logger.error("Failed to insert invoice: %s", e)
        raise
    finally:
        cur.close()
        put_db_connection(conn)


def fetch_all_invoices() -> pd.DataFrame:
    """Fetch all invoices ordered by creation date (descending).

    Returns:
        DataFrame with invoice data.
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql(
            """SELECT id, invoice_number, date, vendor_name, total_amount,
                      currency, source_file, created_at
               FROM invoices ORDER BY created_at DESC""",
            conn,
        )
        return df
    except Exception as e:
        logger.error("Failed to fetch invoices: %s", e)
        raise
    finally:
        put_db_connection(conn)


# ── Semantic Search ──────────────────────────────────────────────────────

def search_similar(
    query: str,
    vendor_filter: Optional[str] = None,
    top_k: int = 5,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    amount_min: Optional[float] = None,
    amount_max: Optional[float] = None,
    keyword_filter: Optional[str] = None,
) -> List[Tuple]:
    """Hybrid semantic + keyword search over invoices.

    Uses pgvector cosine similarity and PostgreSQL full‑text search on raw_text.

    Args:
        query: Natural language search query (embedded via mxbai-embed-large).
        vendor_filter: Optional ILIKE pattern for vendor name.
        top_k: Number of results to return (1-100).
        date_from: Optional start date filter (YYYY-MM-DD).
        date_to: Optional end date filter (YYYY-MM-DD).
        amount_min: Optional minimum total_amount filter.
        amount_max: Optional maximum total_amount filter.
        keyword_filter: Optional keyword/phrase for full‑text search on raw_text.
                        When provided, results must match BOTH the semantic query
                        AND contain the keyword in the OCR text.

    Returns:
        List of tuples: (id, invoice_number, date, vendor_name, total_amount,
                        currency, similarity)
    """
    query_vec = get_embedding(query)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Build WHERE clauses dynamically for structured filters
        conditions = ["embedding IS NOT NULL"]
        params: List[Any] = [query_vec]

        if vendor_filter:
            conditions.append("vendor_name ILIKE %s")
            params.append(f"%{vendor_filter}%")

        if date_from:
            conditions.append("date >= %s")
            params.append(date_from)

        if date_to:
            conditions.append("date <= %s")
            params.append(date_to)

        if amount_min:
            conditions.append("total_amount::numeric >= %s")
            params.append(amount_min)

        if amount_max:
            conditions.append("total_amount::numeric <= %s")
            params.append(amount_max)

        if keyword_filter:
            conditions.append(
                "to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"
            )
            params.append(keyword_filter)

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT id, invoice_number, date, vendor_name, total_amount, currency,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM invoices
            WHERE {where_clause}
            ORDER BY similarity DESC
            LIMIT %s
        """
        params.append(top_k)

        cur.execute(sql, params)
        results = cur.fetchall()
        logger.info("Semantic search returned %d results for query: %s", len(results), query[:50])
        return results if results else []
    except Exception as e:
        logger.error("Semantic search failed: %s", e)
        raise
    finally:
        cur.close()
        put_db_connection(conn)