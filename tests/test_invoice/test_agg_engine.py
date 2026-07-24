"""Unit tests for the aggregation engine module."""

import sys
from pathlib import Path

# Ensure the package is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.invoice.agg_engine import (
    _validate_date,
    _build_where,
    _format_result,
)


class TestValidateDate:
    """Tests for _validate_date()."""

    def test_valid_date(self):
        assert _validate_date("2024-07-20") is True

    def test_invalid_date(self):
        assert _validate_date("not-a-date") is False

    def test_empty_string(self):
        assert _validate_date("") is False

    def test_wrong_format(self):
        assert _validate_date("07/20/2024") is False


class TestBuildWhere:
    """Tests for _build_where()."""

    def test_no_filters(self):
        where, params = _build_where()
        assert where == "TRUE"
        assert params == []

    def test_vendor_filter(self):
        where, params = _build_where(vendor_filter="Alibaba")
        assert "vendor_name ILIKE %s" in where
        assert params == ["%Alibaba%"]

    def test_date_from(self):
        where, params = _build_where(date_from="2024-01-01")
        assert "date >= %s" in where
        assert params == ["2024-01-01"]

    def test_date_to(self):
        where, params = _build_where(date_to="2024-12-31")
        assert "date <= %s" in where
        assert params == ["2024-12-31"]

    def test_all_filters(self):
        where, params = _build_where(
            vendor_filter="Test", date_from="2024-01-01", date_to="2024-12-31"
        )
        assert "vendor_name ILIKE %s" in where
        assert "date >= %s" in where
        assert "date <= %s" in where
        assert len(params) == 3

    def test_invalid_date_ignored(self):
        where, params = _build_where(date_from="bad-date")
        assert "date" not in where
        assert params == []


class TestFormatResult:
    """Tests for _format_result()."""

    def test_empty_rows(self):
        result = _format_result([], ["col1", "col2"], "test query")
        assert "No matching invoices" in result

    def test_single_row_single_column(self):
        rows = [("INV-001",)]
        cols = ["invoice_number"]
        result = _format_result(rows, cols, "test query")
        assert "INV-001" in result
        assert "invoice_number" in result

    def test_multiple_rows(self):
        rows = [("INV-001", "100.00"), ("INV-002", "200.00")]
        cols = ["invoice_number", "amount"]
        result = _format_result(rows, cols, "test query")
        assert "INV-001" in result
        assert "INV-002" in result
        assert "2 results" in result

    def test_null_values(self):
        rows = [(None,)]
        cols = ["amount"]
        result = _format_result(rows, cols, "test query")
        assert "N/A" in result

    def test_truncated_results(self):
        rows = [(f"INV-{i:03d}",) for i in range(15)]
        cols = ["invoice_number"]
        result = _format_result(rows, cols, "test query")
        assert "more rows" in result
        assert "15 results" in result