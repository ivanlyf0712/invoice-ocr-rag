.PHONY: install install-dev serve reset test clean

# ═══════════════════════════════════════════════════════════════
# Install the package in development mode
# ═══════════════════════════════════════════════════════════════
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ═══════════════════════════════════════════════════════════════
# Start the Streamlit app
# ═══════════════════════════════════════════════════════════════
serve:
	streamlit run src/invoice/app.py --server.fileWatcherType none

# ═══════════════════════════════════════════════════════════════
# Run the OCR pipeline
# ═══════════════════════════════════════════════════════════════
pipeline:
	python -m src.invoice.pipeline -f $(file)

pipeline-dir:
	python -m src.invoice.pipeline -d $(dir)

ocr-only:
	python -m src.invoice.pipeline -f $(file) --ocr-only

# ═══════════════════════════════════════════════════════════════
# Database reset
# ═══════════════════════════════════════════════════════════════
reset:
	./scripts/reset.sh -all

reset-clear:
	./scripts/reset.sh -clear

embed:
	./scripts/reset.sh -embed

# ═══════════════════════════════════════════════════════════════
# Run tests
# ═══════════════════════════════════════════════════════════════
test:
	pytest -v

test-cov:
	pytest --cov=src --cov-report=term --cov-report=html

# ═══════════════════════════════════════════════════════════════
# Clean up Python artifacts
# ═══════════════════════════════════════════════════════════════
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleaned build artifacts"