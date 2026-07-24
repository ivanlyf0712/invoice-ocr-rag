#!/bin/bash
# start_server.sh — Launch the llama-server OCR backend (optionally with Streamlit).
# Usage:
#   ./scripts/start_server.sh                  Start OCR server only
#   ./scripts/start_server.sh --with-app       Start OCR server AND Streamlit app
#   Ctrl+C to stop all processes
#
# Kill any existing instance on port 8081 before starting.
# On exit (Ctrl+C), both processes are terminated.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Activate virtual environment ──────────────────────
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  venv/ not found — using system Python"
fi

# ── Config ───────────────────────────────────────────
SERVER_PORT=8081
LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
MODEL="$HOME/uocr/Unlimited-OCR-Q4_K_M.gguf"
MMPROJ="$HOME/uocr/mmproj-Unlimited-OCR-F16.gguf"

# ── Kill existing server on the port ────────────────
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    # Kill OCR server
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        echo "   OCR server (pid $SERVER_PID) stopped."
    fi
    # Kill Streamlit if running
    if [ -n "$ST_PID" ] && kill -0 "$ST_PID" 2>/dev/null; then
        kill "$ST_PID" 2>/dev/null
        echo "   Streamlit (pid $ST_PID) stopped."
    fi
    # Ensure nothing left on the port
    fuser -k ${SERVER_PORT}/tcp 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Check for existing server ───────────────────────
if fuser ${SERVER_PORT}/tcp >/dev/null 2>&1; then
    echo "⚠️  Port ${SERVER_PORT} is in use — killing existing process..."
    fuser -k ${SERVER_PORT}/tcp || true
    sleep 2
fi

# ── Verify model files exist ─────────────────────────
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "❌ llama-server not found at: $LLAMA_SERVER"
    echo "   Build llama.cpp (branch pr-23394) from: https://github.com/ggerganov/llama.cpp/pull/23394"
    echo ""
    echo "   Quick build:"
    echo "     cd ~"
    echo "     git clone https://github.com/ggerganov/llama.cpp.git"
    echo "     cd llama.cpp"
    echo "     git fetch origin pull/23394/head:pr-23394"
    echo "     git checkout pr-23394"
    echo "     mkdir build && cd build"
    echo "     cmake .. -DLLAMA_METAL=ON"
    echo "     cmake --build . --config Release -j4"
    echo "   Expected binary: $LLAMA_SERVER"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "❌ OCR model not found at: $MODEL"
    echo "   Download from: https://huggingface.co/unlimited-ocr"
    exit 1
fi

# ── Start OCR server ─────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Starting llama-server (OCR backend)"
echo "════════════════════════════════════════════"
echo "  Model:    $MODEL"
echo "  Port:     ${SERVER_PORT}"
echo "════════════════════════════════════════════"
echo ""

$LLAMA_SERVER \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --chat-template deepseek-ocr \
    -c 4096 \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --temp 0 \
    --threads 4 \
    --verbose \
    --image-min-tokens 256 \
    --image-max-tokens 1024 \
    --cache-ram 0 \
    --no-cache-prompt \
    --no-kv-offload &
SERVER_PID=$!

echo "✅ llama-server started (pid $SERVER_PID)"
echo ""

# ── Optional: Start Streamlit ───────────────────────
if [ "${1:-}" = "--with-app" ] || [ "${1:-}" = "-a" ]; then
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Starting Streamlit app"
    echo "════════════════════════════════════════════"
    echo ""
    sleep 2  # give server a moment to initialize
    streamlit run src/invoice/app.py --server.fileWatcherType none &
    ST_PID=$!
    echo "✅ Streamlit started (pid $ST_PID)"
    echo ""
    echo "🌐 Open: http://localhost:8501"
fi

echo "Press Ctrl+C to stop all services."
echo ""

# ── Wait for any child process to exit ──────────────
wait