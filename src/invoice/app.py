#!/usr/bin/env python3
"""Streamlit UI for Invoice OCR & RAG – full pipeline with RAG answering.

Usage:
  streamlit run src/invoice/app.py
"""

import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from src.core.classifier import classify_hybrid
from src.core.config import OLLAMA_URL, RAG_MODEL
from src.core.db import (
    get_db_connection,
    put_db_connection,
    insert_invoice,
    update_embedding,
    fetch_all_invoices,
    search_similar,
)
from src.core.extraction import text_to_json, text_to_json_fallback, is_likely_fake
from src.core.ocr import run_ocr
from src.core.pdf import pdf_to_images_bytes
from src.core.ocr_focr import run_ocr_focr, extract_text_from_focr_output
from src.invoice.agg_engine import handle_aggregation, generate_aggregation_sql, _run_sql

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# PATH A: Aggregation (SQL) pipeline
# ──────────────────────────────────────────────────

def run_path_a(
    query: str,
    vendor_filter: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    answer_placeholder: Any,
    classification: Dict[str, Any],
    t_classify: float,
    t_overall: float,
    keyword_filter: Optional[str] = None,
    top_k: int = 5,
    amount_min: Optional[str] = None,
    amount_max: Optional[str] = None,
) -> None:
    """Execute the Path A (aggregation SQL) pipeline.

    Args:
        query: The user's natural language query.
        vendor_filter: Vendor name filter.
        date_from: Start date filter.
        date_to: End date filter.
        answer_placeholder: Streamlit placeholder for the answer.
        classification: Query classification result.
        t_classify: Time when classification finished.
        t_overall: Time when processing started.
        keyword_filter: Keyword filter (unused in Path A but kept for interface consistency).
        top_k: Number of results (unused in Path A).
        amount_min: Minimum amount (unused in Path A).
        amount_max: Maximum amount (unused in Path A).
    """
    st.session_state["search_path"] = "A"

    try:
        with st.spinner("📊 Running aggregation query..."):
            result = generate_aggregation_sql(
                query,
                vendor_filter=vendor_filter if vendor_filter else None,
                date_from=date_from if date_from else None,
                date_to=date_to if date_to else None,
            )
            if result:
                sql, params, desc = result
                raw_rows, raw_cols = _run_sql(sql, params)
                st.session_state["sql_raw_rows"] = raw_rows
                st.session_state["sql_raw_cols"] = raw_cols
                st.session_state["sql_raw_sql"] = sql
                st.session_state["sql_raw_params"] = params
                st.session_state["sql_desc"] = desc
        t_sql = time.time()

        with st.spinner("🤖 Rephrasing result..."):
            answer = handle_aggregation(
                query,
                vendor_filter=vendor_filter if vendor_filter else None,
                date_from=date_from if date_from else None,
                date_to=date_to if date_to else None,
            )
        t_total = time.time()

        if answer:
            timing_breakdown = (
                f"<sub>🧠 Intent: {t_classify - t_overall:.1f}s · "
                f"📊 SQL: {t_sql - t_classify:.1f}s · "
                f"🤖 Rephrase: {t_total - t_sql:.1f}s · "
                f"🕐 Total: {t_total - t_overall:.1f}s</sub>"
            )
            answer_placeholder.markdown(
                f"✅ {answer}\n\n{timing_breakdown}",
                unsafe_allow_html=True,
            )
            st.session_state["current_answer"] = answer
            st.session_state["search_history"].insert(
                0,
                {
                    "query": query,
                    "answer": answer,
                    "path": "A (SQL)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        else:
            answer_placeholder.info("No predefined SQL template matched. Trying semantic search...")
            run_path_b(
                query, vendor_filter, keyword_filter, top_k,
                date_from, date_to, amount_min, amount_max,
                answer_placeholder, classification, t_classify, t_overall,
            )
    except Exception as e:
        logger.exception("Aggregation pipeline failed")
        answer_placeholder.error(f"Aggregation error: {e}")
        # Fall back to semantic search on error
        run_path_b(
            query, vendor_filter, keyword_filter, top_k,
            date_from, date_to, amount_min, amount_max,
            answer_placeholder, classification, t_classify, t_overall,
        )


# ──────────────────────────────────────────────────
# PATH B: Semantic search + RAG pipeline
# ──────────────────────────────────────────────────

def run_path_b(
    query: str,
    vendor_filter: Optional[str],
    keyword_filter: Optional[str],
    top_k: int,
    date_from: Optional[str],
    date_to: Optional[str],
    amount_min: Optional[str],
    amount_max: Optional[str],
    answer_placeholder: Any,
    classification: Dict[str, Any],
    t_classify: float,
    t_overall: float,
) -> None:
    """Execute the Path B (semantic search + RAG) pipeline.

    Args:
        query: The user's natural language query.
        vendor_filter: Vendor name filter.
        keyword_filter: Keyword filter for full-text search.
        top_k: Number of results.
        date_from: Start date filter.
        date_to: End date filter.
        amount_min: Minimum amount filter.
        amount_max: Maximum amount filter.
        answer_placeholder: Streamlit placeholder for the answer.
        classification: Query classification result.
        t_classify: Time when classification finished.
        t_overall: Time when processing started.
    """
    st.session_state["search_path"] = "B"

    # Step 1: Search
    with st.spinner("🔍 Searching invoices..."):
        results = search_similar(
            query,
            vendor_filter=vendor_filter if vendor_filter else None,
            top_k=top_k,
            date_from=date_from if date_from else None,
            date_to=date_to if date_to else None,
            amount_min=amount_min if amount_min else None,
            amount_max=amount_max if amount_max else None,
            keyword_filter=keyword_filter if keyword_filter else None,
        )
        st.session_state["search_results"] = results
    t_search = time.time()

    # Step 2: Generate answer from top 3
    if results:
        MAX_CHARS_PER_INVOICE = 2500
        context_parts = []
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            for r in results[:3]:
                cur.execute("SELECT raw_text FROM invoices WHERE id = %s", (r[0],))
                raw_row = cur.fetchone()
                if raw_row and raw_row[0]:
                    raw = raw_row[0].strip()
                    if len(raw) > MAX_CHARS_PER_INVOICE:
                        raw = raw[:MAX_CHARS_PER_INVOICE] + "\n... [truncated]"
                    context_parts.append(raw)
            cur.close()
        finally:
            put_db_connection(conn)

        if context_parts:
            context = "\n\n".join(context_parts)
            rag_prompt = f"""請根據以下發票內容回答問題。如果無法回答，請說「資料不足」。
問題：{query}
發票內容：{context}
答案："""

            with st.spinner("🧠 Generating answer..."):
                try:
                    t_gen_start = time.time()
                    resp = requests.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={
                            "model": RAG_MODEL,
                            "prompt": rag_prompt,
                            "stream": False,
                            "options": {"temperature": 0.1, "num_predict": 256},
                        },
                        timeout=120,
                    )
                    t_gen = time.time()
                    t_total = t_gen
                    if resp.status_code == 200:
                        answer = resp.json().get("response", "").strip()
                        if answer:
                            timing_breakdown = (
                                f"<sub>🧠 Intent: {t_classify - t_overall:.1f}s · "
                                f"🔍 Search: {t_search - t_classify:.1f}s · "
                                f"🤖 RAG: {t_gen - t_gen_start:.1f}s · "
                                f"🕐 Total: {t_total - t_overall:.1f}s</sub>"
                            )
                            answer_placeholder.markdown(
                                f"✅ {answer}\n\n{timing_breakdown}",
                                unsafe_allow_html=True,
                            )
                            st.session_state["current_answer"] = answer
                            st.session_state["search_history"].insert(
                                0,
                                {
                                    "query": query,
                                    "answer": answer,
                                    "path": "B (RAG)",
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                },
                            )
                    else:
                        answer_placeholder.error(f"RAG model error (HTTP {resp.status_code})")
                except requests.exceptions.Timeout:
                    answer_placeholder.error("Request timed out.")
                except Exception as e:
                    answer_placeholder.error(f"Unexpected error: {e}")
        else:
            answer_placeholder.warning("No OCR text available for these invoices.")
    else:
        answer_placeholder.info("No matching invoices found.")


# ──────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────

def display_answer() -> None:
    """Display the current answer from session state."""
    current_answer = st.session_state.get("current_answer")
    search_path = st.session_state.get("search_path")
    submitted = st.session_state.get("_just_submitted", False)

    if current_answer and search_path:
        if search_path == "A" and not submitted:
            st.success(current_answer)
            st.caption("⏱ SQL aggregation")


def display_classification() -> None:
    """Display the recognised fields from the classification."""
    classification = st.session_state.get("classification")
    if classification:
        with st.expander("🔎 Recognised Fields", expanded=False):
            fields = []
            if classification.get("vendor"):
                fields.append(f"  vendor = `{classification['vendor']}`")
            if classification.get("date_from"):
                fields.append(f"  date_from = `{classification['date_from']}`")
            if classification.get("date_to"):
                fields.append(f"  date_to = `{classification['date_to']}`")
            fields.append(f"  method = `{classification.get('method', '?')}`")
            fields.append(f"  intent = `{classification.get('intent', '?')}`")
            if classification.get("aggregation_type"):
                fields.append(f"  aggregation_type = `{classification['aggregation_type']}`")
            st.code("\n".join(fields), language=None)


def display_sql_result() -> None:
    """Display the raw SQL result (Path A)."""
    if st.session_state.get("sql_raw_rows") is not None:
        st.divider()

        display_classification()

        # Generated SQL
        sql_text = st.session_state.get("sql_raw_sql")
        if sql_text:
            with st.expander("📝 Generated SQL", expanded=False):
                st.code(sql_text.strip(), language="sql")

        # Result table
        raw_rows = st.session_state["sql_raw_rows"]
        raw_cols = st.session_state["sql_raw_cols"]
        desc = st.session_state.get("sql_desc", "SQL result")
        st.caption(f"📊 {desc} — {len(raw_rows)} row(s)")
        if raw_cols:
            df_sql = pd.DataFrame(raw_rows, columns=raw_cols)
            st.dataframe(df_sql, width="stretch")


def display_search_results() -> None:
    """Display the semantic search results (Path B)."""
    results = st.session_state.get("search_results")
    if results is not None:
        if results:
            st.divider()
            st.caption(f"📋 Top {len(results)} matching invoices")

            # Fetch raw_text for expanders
            if "results_raw_map" not in st.session_state or st.session_state.get("results_raw_map_id") != id(results):
                conn = get_db_connection()
                try:
                    cur = conn.cursor()
                    result_ids = [r[0] for r in results]
                    cur.execute(
                        "SELECT id, raw_text FROM invoices WHERE id = ANY(%s)",
                        (result_ids,),
                    )
                    st.session_state["results_raw_map"] = {
                        r[0]: (r[1] or "") for r in cur.fetchall()
                    }
                    st.session_state["results_raw_map_id"] = id(results)
                    cur.close()
                finally:
                    put_db_connection(conn)

            raw_map = st.session_state["results_raw_map"]

            df_res = pd.DataFrame(
                [r[1:] for r in results],
                columns=[
                    "Invoice Number",
                    "Date",
                    "Vendor",
                    "Amount",
                    "Currency",
                    "Similarity",
                ],
            )
            df_res["Similarity"] = df_res["Similarity"].apply(lambda x: f"{x:.4f}")
            df_res["OCR Snippet"] = [
                (
                    raw_map.get(r[0], "")[:100] + "..."
                    if len(raw_map.get(r[0], "")) > 100
                    else raw_map.get(r[0], "")
                )
                for r in results
            ]
            df_res["ID"] = [r[0] for r in results]

            st.dataframe(
                df_res[
                    [
                        "ID",
                        "Invoice Number",
                        "Date",
                        "Vendor",
                        "Amount",
                        "Currency",
                        "Similarity",
                        "OCR Snippet",
                    ]
                ],
                width="stretch",
                column_config={
                    "OCR Snippet": st.column_config.TextColumn("OCR Snippet", width="medium"),
                    "Similarity": st.column_config.TextColumn("Similarity", width="small"),
                },
            )

            for i, r in enumerate(results):
                raw = raw_map.get(r[0], "(no OCR text)")
                with st.expander(f"📄 #{i+1} — {r[1]} | {r[3]} | ${r[4]}"):
                    st.text(raw[:3000] + ("\n\n... [truncated]" if len(raw) > 3000 else ""))
        else:
            st.info("No matching invoices found.")


def display_history() -> None:
    """Display the search history."""
    if st.session_state["search_history"]:
        st.divider()
        with st.expander(f"📜 Search History ({len(st.session_state['search_history'])})"):
            for idx, entry in enumerate(st.session_state["search_history"]):
                path_tag = entry.get("path", "")
                st.markdown(
                    f"**Q{idx+1}** [{path_tag}] {entry['query']} — *{entry['timestamp']}*"
                )
                st.markdown(f"> {entry['answer']}")
                st.markdown("---")
            if st.button("🗑️ Clear History"):
                st.session_state["search_history"] = []
                st.session_state["current_answer"] = None
                st.rerun()


# ──────────────────────────────────────────────────
# Upload processing
# ──────────────────────────────────────────────────

def process_upload(uploaded_file: Any, use_focr: bool = False) -> None:
    """Process an uploaded file (image or PDF) through the OCR pipeline.

    Args:
        uploaded_file: A Streamlit uploaded file object.
        use_focr: If True, use FOCR backend instead of llama.cpp for OCR.
    """
    with st.spinner("Running OCR pipeline (this may take ~15 seconds per page)..."):
        if uploaded_file.name.lower().endswith(".pdf"):
            pdf_bytes = uploaded_file.getvalue()

            if use_focr:
                t0 = time.time()
                use_multipage = bool(st.session_state.get("focr_multipage", False))
                if use_multipage:
                    # ── FOCR native multipage PDF ───────────────────────
                    tmp_pdf = f"/tmp/{uploaded_file.name}"
                    try:
                        with open(tmp_pdf, "wb") as f:
                            f.write(pdf_bytes)
                        st.write("📄 Using FOCR multipage PDF...")
                        raw_text = run_ocr_focr(tmp_pdf, multi_page=True)
                        raw_text = extract_text_from_focr_output(raw_text)
                    except subprocess.CalledProcessError:
                        st.warning("Native FOCR multipage failed; falling back to page-by-page rasterization.")
                        pages = pdf_to_images_bytes(pdf_bytes, uploaded_file.name)
                        st.write(f"PDF has {len(pages)} pages. Processing with FOCR...")
                        all_texts = []
                        for idx, (img_path, source_name) in enumerate(pages, 1):
                            st.write(f"  📄 FOCR page {idx}/{len(pages)}: {source_name}")
                            txt = run_ocr_focr(img_path, multi_page=False)
                            txt = extract_text_from_focr_output(txt)
                            all_texts.append(txt)
                            os.remove(img_path)
                        raw_text = "\n\n".join(all_texts)
                    finally:
                        if os.path.exists(tmp_pdf):
                            os.unlink(tmp_pdf)
                else:
                    # ── FOCR page-by-page via rasterized images ─────────
                    pages = pdf_to_images_bytes(pdf_bytes, uploaded_file.name)
                    st.write(f"PDF has {len(pages)} pages. Processing with FOCR...")
                    all_texts = []
                    for idx, (img_path, source_name) in enumerate(pages, 1):
                        st.write(f"  📄 FOCR page {idx}/{len(pages)}: {source_name}")
                        txt = run_ocr_focr(img_path, multi_page=False)
                        txt = extract_text_from_focr_output(txt)
                        all_texts.append(txt)
                        os.remove(img_path)
                    raw_text = "\n\n".join(all_texts)
                t1 = time.time()
                st.write(f"⏱ OCR: {t1 - t0:.1f}s")

                data = text_to_json(raw_text)
                if data and is_likely_fake(data):
                    data = text_to_json_fallback(raw_text)
                if data is None or is_likely_fake(data):
                    data = {}
                else:
                    st.json(data)

                new_id = insert_invoice(data, raw_text, uploaded_file.name)
                st.success(f"Inserted with ID {new_id}")

                with st.spinner("Generating embedding..."):
                    update_embedding(new_id)
                st.success("Embedding generated.")
                st.write(f"🕐 Total: {time.time() - t0:.1f}s")
            else:
                # ── Legacy multipage: convert PDF → images → OCR per page ──
                pages = pdf_to_images_bytes(pdf_bytes, uploaded_file.name)
                st.write(f"PDF has {len(pages)} pages. Processing each page...")
                for img_path, source_name in pages:
                    raw_text = run_ocr(img_path)
                    data = text_to_json(raw_text)
                    if data and is_likely_fake(data):
                        data = text_to_json_fallback(raw_text)
                    if data is None or is_likely_fake(data):
                        data = {}
                    new_id = insert_invoice(data, raw_text, source_name)
                    update_embedding(new_id)
                    os.remove(img_path)
                st.success(f"Processed {len(pages)} pages from PDF.")
        else:
            tmp_path = f"/tmp/{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            t0 = time.time()

            if use_focr:
                st.write("✨ Using FOCR backend...")
                try:
                    raw_text = run_ocr_focr(tmp_path)
                    raw_text = extract_text_from_focr_output(raw_text)
                except subprocess.CalledProcessError as e:
                    st.error(f"FOCR failed with exit code {e.returncode}")
                    if e.stderr:
                        st.code(e.stderr, language="text")
                    raise
            else:
                raw_text = run_ocr(tmp_path)

            t1 = time.time()
            st.write(f"⏱ OCR: {t1 - t0:.1f}s")

            data = text_to_json(raw_text)
            if data and is_likely_fake(data):
                st.warning("First attempt returned placeholder data – retrying with stricter prompt...")
                data = text_to_json_fallback(raw_text)

            t2 = time.time()
            st.write(f"⏱ JSON parse: {t2 - t1:.1f}s")

            if data is None or is_likely_fake(data):
                st.error("Could not extract valid JSON – raw text will be stored.")
                data = {}
            else:
                st.json(data)

            new_id = insert_invoice(data, raw_text, uploaded_file.name)
            st.success(f"Inserted with ID {new_id}")

            with st.spinner("Generating embedding..."):
                update_embedding(new_id)
            st.success("Embedding generated.")

            st.write(f"🕐 Total: {time.time() - t0:.1f}s")
            os.remove(tmp_path)


# ──────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(page_title="Invoice OCR & RAG", layout="wide")
    st.title("📄 Unlimited‑OCR & RAG Dashboard")

    # Default backend for this branch is R-SWA
    os.environ["OCR_BACKEND"] = "rswa"
    use_focr = False

    tab1, tab2, tab3 = st.tabs(["📋 View Database", "📤 Upload Invoice", "🔍 Search"])

    with tab1:
        st.subheader("Invoices Table")

        # DB Statistics
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), COUNT(embedding), COUNT(raw_text) FROM invoices")
            total, with_emb, with_raw = cur.fetchone()
            cur.close()
        finally:
            put_db_connection(conn)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total Invoices", total)
        with col_s2:
            emb_pct = f"{with_emb / total * 100:.0f}%" if total > 0 else "0%"
            st.metric("With Embeddings", f"{with_emb} ({emb_pct})")
        with col_s3:
            raw_pct = f"{with_raw / total * 100:.0f}%" if total > 0 else "0%"
            st.metric("With OCR Text", f"{with_raw} ({raw_pct})")
        with col_s4:
            conn_idx = get_db_connection()
            try:
                cur_idx = conn_idx.cursor()
                cur_idx.execute("""
                    SELECT indexname FROM pg_indexes
                    WHERE tablename = 'invoices'
                    AND (indexname LIKE '%hnsw%' OR indexname LIKE '%trgm%')
                """)
                idx_count = len(cur_idx.fetchall())
                cur_idx.close()
            finally:
                put_db_connection(conn_idx)
            st.metric("Indexes Active", idx_count)

        df = fetch_all_invoices()
        if not df.empty:
            # OCR preview
            conn_raw = get_db_connection()
            try:
                cur_raw = conn_raw.cursor()
                cur_raw.execute("SELECT id, raw_text FROM invoices ORDER BY created_at DESC")
                raw_map = {r[0]: (r[1] or "")[:120] for r in cur_raw.fetchall()}
                cur_raw.close()
            finally:
                put_db_connection(conn_raw)

            df["OCR Preview"] = df["id"].map(
                lambda x: (
                    raw_map.get(x, "") + "..."
                    if len(raw_map.get(x, "")) >= 120
                    else raw_map.get(x, "")
                )
            )

            # Embedding status
            conn_emb = get_db_connection()
            try:
                cur_emb = conn_emb.cursor()
                cur_emb.execute("SELECT id FROM invoices WHERE embedding IS NOT NULL")
                emb_ids = {r[0] for r in cur_emb.fetchall()}
                cur_emb.close()
            finally:
                put_db_connection(conn_emb)
            df["Embedded"] = df["id"].apply(lambda x: "✅" if x in emb_ids else "❌")

            display_cols = [
                "id", "invoice_number", "date", "vendor_name", "total_amount",
                "currency", "Embedded", "OCR Preview", "source_file", "created_at",
            ]
            df = df[[c for c in display_cols if c in df.columns]]

            st.dataframe(
                df,
                width="stretch",
                column_config={
                    "OCR Preview": st.column_config.TextColumn(
                        "OCR Preview",
                        help="First 120 chars of raw OCR text. Hover to read.",
                        width="medium",
                    ),
                    "Embedded": st.column_config.TextColumn(
                        "Embedded",
                        help="✅ = embedding generated | ❌ = needs embedding",
                        width="small",
                    ),
                },
            )
            st.caption("💡 Hover over the **OCR Preview** column to see the beginning of each invoice's OCR text.")
            st.caption("💡 **Embedded** = ❌ means the row has no embedding yet. Run `python -m src.invoice.pipeline` to generate them.")
        else:
            st.info("No invoices in database. Upload one via the Upload tab.")

    with tab2:
        st.subheader("Upload Invoice Image or PDF")
        uploaded_file = st.file_uploader(
            "Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"]
        )

        col_toggle, col_info = st.columns([1, 3])
        with col_toggle:
            use_focr = st.toggle(
                "FOCR",
                value=False,
                help="Toggle to use Franken OCR (FOCR) instead of R-SWA.",
            )
        with col_info:
            if use_focr:
                st.caption("✨ Using **FOCR** backend")
            else:
                st.caption("⚙️ Using **R-SWA** backend")

        # Sync toggle to backend used by run_ocr() in this run
        os.environ["OCR_BACKEND"] = "focr" if use_focr else "rswa"

        if use_focr:
            if "focr_multipage" not in st.session_state:
                st.session_state["focr_multipage"] = False
            st.checkbox(
                "🧩 FOCR multipage",
                key="focr_multipage",
                value=False,
                help="When enabled, pass the PDF directly to FOCR with --multi-page. "
                     "When disabled, rasterize PDF pages to images and OCR one-by-one.",
            )

        if uploaded_file is not None:
            st.write(f"File: **{uploaded_file.name}**")
            if st.button("Process Invoice"):
                process_upload(uploaded_file, use_focr=use_focr)

    with tab3:
        st.subheader("Semantic Search over Invoices")

        # Initialize session state
        if "search_history" not in st.session_state:
            st.session_state["search_history"] = []
        if "current_answer" not in st.session_state:
            st.session_state["current_answer"] = None
        if "sql_raw_rows" not in st.session_state:
            st.session_state["sql_raw_rows"] = None
        if "sql_raw_cols" not in st.session_state:
            st.session_state["sql_raw_cols"] = None
        if "sql_raw_sql" not in st.session_state:
            st.session_state["sql_raw_sql"] = None
        if "sql_raw_params" not in st.session_state:
            st.session_state["sql_raw_params"] = None
        if "classification" not in st.session_state:
            st.session_state["classification"] = None
        if "search_path" not in st.session_state:
            st.session_state["search_path"] = None

        # Search bar
        st.markdown("<br>", unsafe_allow_html=True)
        col_center = st.columns([1, 8, 1])
        with col_center[1]:
            query = st.text_input(
                "Invoice Query",
                placeholder="Ask anything... e.g., What is the total amount for Alibaba in 2024?",
                label_visibility="collapsed",
                key="main_query",
            )

        col_btn = st.columns([5, 1, 5])
        with col_btn[1]:
            submitted = st.button("🔍 Answer", width="stretch", type="primary")

        # Filters
        with st.expander("⚙️ Filters", expanded=False):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                vendor_filter = st.text_input("Vendor", placeholder="e.g., Alibaba")
            with col_f2:
                keyword_filter = st.text_input(
                    "Keyword in OCR text",
                    placeholder="e.g., shipping",
                    help="PostgreSQL full‑text search on raw_text.",
                )
            with col_f3:
                top_k = st.slider("Results", 1, 20, 5)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                date_from = st.text_input("Date from (YYYY-MM-DD)", placeholder="2024-01-01")
            with col_d2:
                date_to = st.text_input("Date to (YYYY-MM-DD)", placeholder="2024-12-31")

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                amount_min = st.text_input("Amount min", placeholder="0")
            with col_a2:
                amount_max = st.text_input("Amount max", placeholder="10000")

        # Handle submission
        if submitted:
            if not query:
                st.warning("Please enter a query.")
            else:
                st.session_state["_just_submitted"] = True
                st.session_state["current_answer"] = None
                st.session_state["sql_raw_rows"] = None
                st.session_state["sql_raw_cols"] = None
                st.session_state["search_results"] = None
                st.session_state["search_query"] = query

                t_overall = time.time()

                # Step 1: Classify
                with st.spinner("🧠 Identifying intent..."):
                    classification = classify_hybrid(query)
                t_classify = time.time()
                st.session_state["classification"] = classification
                st.write(
                    f"⏱ Intent classification: {t_classify - t_overall:.1f}s "
                    f"({classification.get('method', '?')})"
                )

                # Auto-fill filters from classification
                if not vendor_filter and classification.get("vendor"):
                    vendor_filter = classification["vendor"]
                if not date_from and classification.get("date_from"):
                    date_from = classification["date_from"]
                if not date_to and classification.get("date_to"):
                    date_to = classification["date_to"]

                answer_placeholder = st.empty()

                # Path A or Path B
                if classification.get("intent") == "aggregation":
                    run_path_a(
                        query, vendor_filter, date_from, date_to,
                        answer_placeholder, classification, t_classify, t_overall,
                        keyword_filter, top_k, amount_min, amount_max,
                    )
                else:
                    run_path_b(
                        query, vendor_filter, keyword_filter, top_k,
                        date_from, date_to, amount_min, amount_max,
                        answer_placeholder, classification, t_classify, t_overall,
                    )

                st.session_state["_just_submitted"] = False
        else:
            st.session_state["_just_submitted"] = False

        # Display results
        display_answer()
        display_sql_result()
        display_search_results()

        # How Search Works + History
        with st.expander("🔍 How Search Works"):
            st.markdown("""
            ### Two‑Path Query Router

            **Path A — Aggregation (SQL)** for questions with *total, sum, average, count, highest, lowest, summarize*:
            - Query → SQL (SUM/AVG/COUNT/GROUP BY) → exact result → LLM rephrases

            **Path B — Semantic Search + RAG** for open‑ended questions:
            - Query → embedding (mxbai-embed-large, 1024‑dim) → HNSW cosine similarity → top‑3 OCR text → LLM answers

            ### Indexes

            | Index | Accelerates |
            |---|---|
            | **HNSW** on `embedding` | Cosine similarity, sub‑ms even at 100K+ rows |
            | **GIN trigram** on `vendor_name` | ILIKE vendor filters |
            | **Full‑text** on `raw_text` | Keyword search in OCR text |
            """)

        display_history()


if __name__ == "__main__":
    main()