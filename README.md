# OCR Platform — Invoice OCR & RAG

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-grade invoice OCR and Retrieval-Augmented Generation (RAG) platform powered by [Unlimited-OCR](https://huggingface.co/unlimited-ocr) and Ollama. Features a hybrid query router that intelligently dispatches questions to either a SQL aggregation engine or a semantic search pipeline.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│       Query Classifier              │
│  (regex-based intent detection)     │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌────────────┐   ┌──────────────────┐
│ Path A     │   │ Path B           │
│ SQL Agg.   │   │ Semantic + RAG   │
│ ────────── │   │ ───────────────  │
│ Natural    │   │ Embedding query  │
│ language → │   │ → pgvector       │
│ SQL →      │   │ cosine sim.      │
│ PostgreSQL │   │ → top-3 OCR text │
│ → LLM      │   │ → LLM answers    │
│ rephrase   │   │                  │
└────────────┘   └──────────────────┘
    │                     │
    └──────────┬──────────┘
               ▼
       Streamlit UI
```

### Query Routing

| Intent | Path | Method |
|--------|------|--------|
| Aggregation (sum, avg, count, highest, lowest) | **A** | SQL → PostgreSQL → LLM rephrase |
| Semantic / open-ended | **B** | Embedding → HNSW cosine similarity → RAG |

---

## Prerequisites

- **Python 3.9+**
- **PostgreSQL 16** with pgvector extension (provided via Docker)
- **Ollama** (provided via Docker)
- **llama.cpp** — [branch `pr-23394`](https://github.com/ggerganov/llama.cpp/pull/23394) (OCR backend)
- **Unlimited-OCR model files** (GGUF format)

---

## Quick Start

### 1. Clone and Install

```bash
git clone <repo-url> ocr-platform
cd ocr-platform

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install the package and dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to match your setup — defaults work for local development
```

### 3. Start Infrastructure Services

```bash
# Start PostgreSQL
docker compose up -d postgres

# Start Ollama
docker compose up -d ollama

# Pull required Ollama models
ollama pull qwen2.5:1.5b
ollama pull mxbai-embed-large
```

### 4. Build and Start the OCR Server

The OCR engine requires [llama.cpp](https://github.com/ggerganov/llama.cpp) built from branch `pr-23394`:

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git fetch origin pull/23394/head:pr-23394
git checkout pr-23394
mkdir build && cd build
cmake .. -DLLAMA_METAL=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)
```

Download the OCR model files from [Hugging Face](https://huggingface.co/unlimited-ocr):

```bash
# Place them at the paths expected by .env:
#   ~/uocr/Unlimited-OCR-Q4_K_M.gguf
#   ~/uocr/mmproj-Unlimited-OCR-F16.gguf
```

Start the OCR server:

```bash
./scripts/start_server.sh
```

### 5. Launch the Application

```bash
# Start the Streamlit dashboard
streamlit run src/invoice/app.py --server.fileWatcherType none
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Alternatively, start both services together:

```bash
./scripts/start_server.sh --with-app
```

---

## Usage

### Streamlit Dashboard (Web UI)

| Tab | Function |
|-----|----------|
| **View Database** | Browse invoices, check embedding status, view OCR text |
| **Upload Invoice** | Upload images (JPG, PNG) or PDFs for OCR processing |
| **Search** | Natural-language query with filters (vendor, date, amount, keyword) |

### CLI Pipeline

```bash
# Process a single invoice image
python -m src.invoice.pipeline -f samples/invoice/sample_invoice.jpg

# Process a PDF
python -m src.invoice.pipeline -f samples/invoice/sample_invoice.pdf

# Batch process a directory
python -m src.invoice.pipeline -d ./samples/invoice/

# OCR only (no database insertion)
python -m src.invoice.pipeline -f invoice.jpg --ocr-only
```

### Available Commands (Make)

```bash
make serve          # Start Streamlit UI
make pipeline file=invoice.jpg   # Process single file
make test           # Run all tests
make test-cov       # Run tests with coverage report
make clean          # Remove build artifacts
```

---

## Project Structure

```
ocr-platform/
├── src/
│   ├── core/                    # Core modules
│   │   ├── config.py            # Environment configuration
│   │   ├── db.py                # PostgreSQL connection pool & CRUD
│   │   ├── ocr.py               # OCR engine (server + CLI modes)
│   │   ├── pdf.py               # PDF-to-image conversion
│   │   ├── embedding.py         # Embedding generation (mxbai-embed-large)
│   │   ├── extraction.py        # JSON extraction from OCR text
│   │   └── classifier.py        # Query classification (intent + filters)
│   └── invoice/
│       ├── agg_engine.py        # SQL aggregation engine (Path A)
│       ├── pipeline.py          # CLI pipeline entry point
│       ├── app.py               # Streamlit web application
│       └── sql/
│           └── init_invoice.sql # Database schema
├── scripts/
│   ├── start_server.sh          # OCR server + optional Streamlit launcher
│   ├── reset.sh                 # Database maintenance utilities
│   └── embed_update.py          # Embedding update script
├── tests/
│   ├── test_core/
│   │   ├── test_classifier.py   # 33 tests for query classification
│   │   └── test_extraction.py   # 19 tests for data extraction
│   └── test_invoice/
│       └── test_agg_engine.py   # 17 tests for aggregation engine
├── samples/                     # Sample invoices for testing
├── docker-compose.yml           # PostgreSQL + Ollama containers
├── pyproject.toml               # Package configuration
├── Makefile                     # Development commands
└── .env.example                 # Environment template
```

---

## Database Schema

The `invoices` table includes:

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL PRIMARY KEY` | Auto-increment ID |
| `invoice_number` | `VARCHAR(100)` | Invoice number |
| `date` | `VARCHAR(20)` | Invoice date (YYYY-MM-DD) |
| `vendor_name` | `VARCHAR(255)` | Vendor/supplier name |
| `total_amount` | `VARCHAR(50)` | Total amount (kept as string for precision) |
| `currency` | `VARCHAR(10)` | Three-letter currency code |
| `raw_text` | `TEXT` | Raw OCR output text |
| `source_file` | `VARCHAR(500)` | Original source filename |
| `embedding` | `vector(1024)` | mxbai-embed-large embedding vector |
| `created_at` | `TIMESTAMP DEFAULT NOW()` | Record creation time |

Indexes:
- **HNSW** on `embedding` — fast cosine similarity search
- **GIN trigram** on `vendor_name` — ILIKE pattern matching
- **Full-text search** on `raw_text` — keyword search in OCR text

---

## Test Suite

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src --cov-report=term

# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**Current status:** 66/69 tests passing (3 pre-existing minor failures related to CJK keyword matching and null-value formatting).

---

## Configuration Reference

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_MODE` | `server` | OCR backend: `server` or `cli` |
| `OCR_SERVER_URL` | `http://127.0.0.1:8081/v1/chat/completions` | llama-server endpoint |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `TEXT_MODEL` | `qwen2.5:1.5b` | Model for JSON extraction |
| `EMBED_MODEL` | `mxbai-embed-large` | Model for embeddings |
| `RAG_MODEL` | `qwen2.5:1.5b` | Model for RAG answers |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_USER` | `ocr` | PostgreSQL user |
| `DB_PASSWORD` | *(required, no default)* | PostgreSQL password |
| `DB_NAME` | `invoices` | PostgreSQL database |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Ensure the package is installed in editable mode:

```bash
pip install -e .
```

### OCR server connection refused

1. Verify llama-server is running: `curl http://127.0.0.1:8081/v1/chat/completions`
2. Check model paths in `.env`
3. Rebuild llama.cpp from branch `pr-23394` if needed

### PostgreSQL connection issues

```bash
docker compose up -d postgres
docker exec postgres psql -U ocr -d invoices -c "SELECT COUNT(*) FROM invoices;"
```

### Ollama not responding

```bash
docker compose up -d ollama
ollama list  # Should show available models
```

---

## License

MIT License — see the [LICENSE](LICENSE) file for details.


---

## Acknowledgments

- [Unlimited-OCR](https://huggingface.co/unlimited-ocr) — Multimodal OCR model
- [Ollama](https://ollama.ai) — Local LLM runtime
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — C++ LLM inference (branch `pr-23394`)
- [pgvector](https://github.com/pgvector/pgvector) — Vector similarity search for PostgreSQL
- [Streamlit](https://streamlit.io) — Web UI framework
