"""Unit tests for the extraction module."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure the package is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.core.extraction import (
    clean_invoice_data,
    is_likely_fake,
    _normalise_date,
)


class TestNormaliseDate:
    """Tests for _normalise_date()."""

    def test_iso_format(self):
        assert _normalise_date("2024-07-20") == "2024-07-20"

    def test_us_format(self):
        assert _normalise_date("07/20/2024") == "2024-07-20"

    def test_eu_format(self):
        assert _normalise_date("20/07/2024") == "2024-07-20"

    def test_dmy_dash(self):
        assert _normalise_date("20-Jul-2024") == "2024-07-20"

    def test_dmy_space(self):
        assert _normalise_date("20 Jul 2024") == "2024-07-20"

    def test_yyyy_slash(self):
        assert _normalise_date("2024/07/20") == "2024-07-20"

    def test_dot_format(self):
        assert _normalise_date("20.07.2024") == "2024-07-20"

    def test_empty_string(self):
        assert _normalise_date("") == ""

    def test_invalid_date(self):
        assert _normalise_date("not-a-date") == "not-a-date"


class TestCleanInvoiceData:
    """Tests for clean_invoice_data()."""

    def test_clean_string_amount(self):
        data = {
            "invoice_number": "INV-001",
            "date": "2024-07-20",
            "vendor_name": "Test Corp",
            "total_amount": "$1,250.00",
            "currency": "USD",
        }
        result = clean_invoice_data(data)
        assert result["total_amount"] == "1250.00"
        assert result["currency"] == "USD"

    def test_clean_float_amount(self):
        data = {
            "invoice_number": "INV-002",
            "date": "2024-07-21",
            "vendor_name": "Test Corp",
            "total_amount": 1250.0,
            "currency": "USD",
        }
        result = clean_invoice_data(data)
        assert result["total_amount"] == "1250.00"

    def test_clean_int_amount(self):
        data = {
            "invoice_number": "INV-003",
            "date": "2024-07-22",
            "vendor_name": "Test Corp",
            "total_amount": 1250,
            "currency": "USD",
        }
        result = clean_invoice_data(data)
        assert result["total_amount"] == "1250.00"

    def test_clean_dict_amount(self):
        data = {
            "invoice_number": "INV-004",
            "date": "2024-07-23",
            "vendor_name": "Test Corp",
            "total_amount": {"amount": "1250.00", "currency": "USD"},
        }
        result = clean_invoice_data(data)
        assert result["total_amount"] == "1250.00"
        assert result["currency"] == "USD"

    def test_missing_fields(self):
        data = {}
        result = clean_invoice_data(data)
        assert result["invoice_number"] == ""
        assert result["date"] == ""
        assert result["vendor_name"] == ""
        assert result["total_amount"] == ""
        assert result["currency"] == ""

    def test_currency_lowercase(self):
        data = {
            "invoice_number": "INV-005",
            "date": "2024-07-24",
            "vendor_name": "Test Corp",
            "total_amount": "1250.00",
            "currency": "usd",
        }
        result = clean_invoice_data(data)
        assert result["currency"] == "USD"

    def test_date_normalisation(self):
        data = {
            "invoice_number": "INV-006",
            "date": "07/25/2024",
            "vendor_name": "Test Corp",
            "total_amount": "1250.00",
            "currency": "USD",
        }
        result = clean_invoice_data(data)
        assert result["date"] == "2024-07-25"

    def test_amount_with_commas(self):
        data = {
            "invoice_number": "INV-007",
            "date": "2024-07-26",
            "vendor_name": "Test Corp",
            "total_amount": "1,250.00",
            "currency": "USD",
        }
        result = clean_invoice_data(data)
        assert result["total_amount"] == "1250.00"


class TestIsLikelyFake:
    """Tests for is_likely_fake()."""

    def test_none_data(self):
        assert is_likely_fake(None) is True

    def test_valid_data(self):
        data = {
            "invoice_number": "INV-001",
            "date": "2024-07-20",
            "vendor_name": "Test Corp",
            "total_amount": "1250.00",
        }
        assert is_likely_fake(data) is False

    def test_placeholder_value(self):
        data = {
            "invoice_number": "value",
            "date": "2024-07-20",
            "vendor_name": "Test Corp",
            "total_amount": "1250.00",
        }
        assert is_likely_fake(data) is True

    def test_example_value(self):
        data = {
            "invoice_number": "INV-001",
            "date": "2024-07-20",
            "vendor_name": "example",
            "total_amount": "1250.00",
        }
        assert is_likely_fake(data) is True

    def test_xxxx_value(self):
        data = {
            "invoice_number": "xxxx",
            "date": "2024-07-20",
            "vendor_name": "Test Corp",
            "total_amount": "1250.00",
        }
        assert is_likely_fake(data) is True

    def test_empty_data(self):
        data = {
            "invoice_number": "",
            "date": "",
            "vendor_name": "",
            "total_amount": "",
        }
        assert is_likely_fake(data) is False