"""Unit tests for the query classifier module."""

import sys
from pathlib import Path

# Ensure the package is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.core.classifier import (
    is_aggregation_query,
    extract_vendor_from_query,
    extract_date_range_from_query,
)


class TestIsAggregationQuery:
    """Tests for is_aggregation_query()."""

    def test_total_keyword(self):
        assert is_aggregation_query("What is the total amount?") is True

    def test_sum_keyword(self):
        assert is_aggregation_query("Sum of all invoices") is True

    def test_average_keyword(self):
        assert is_aggregation_query("Average invoice amount") is True

    def test_count_keyword(self):
        assert is_aggregation_query("How many invoices do I have?") is True

    def test_list_keyword(self):
        assert is_aggregation_query("List all invoices") is True

    def test_show_keyword(self):
        assert is_aggregation_query("Show me invoices") is True

    def test_semantic_query(self):
        assert is_aggregation_query("What is this invoice about?") is False

    def test_semantic_with_about(self):
        assert is_aggregation_query("Tell me about this invoice") is False

    def test_semantic_with_related(self):
        assert is_aggregation_query("Find invoices related to shipping") is False

    def test_empty_query(self):
        assert is_aggregation_query("") is False

    def test_chinese_aggregation(self):
        assert is_aggregation_query("总金额是多少？") is True

    def test_chinese_count(self):
        assert is_aggregation_query("有几个发票？") is True


class TestExtractVendorFromQuery:
    """Tests for extract_vendor_from_query()."""

    def test_for_vendor(self):
        assert extract_vendor_from_query("Total for Alibaba in 2024") == "Alibaba"

    def test_from_vendor(self):
        assert extract_vendor_from_query("Invoices from Apple Inc") == "Apple Inc"

    def test_to_vendor(self):
        assert extract_vendor_from_query("Payments to Microsoft") == "Microsoft"

    def test_vendor_invoices_pattern(self):
        assert extract_vendor_from_query("Amazon invoices") == "Amazon"

    def test_made_to_vendor(self):
        assert extract_vendor_from_query("Payments made to Google") == "Google"

    def test_no_vendor(self):
        assert extract_vendor_from_query("What is the total amount?") is None

    def test_vendor_with_ampersand(self):
        result = extract_vendor_from_query("Invoices from AT&T")
        assert result is not None
        assert "AT&T" in result

    def test_vendor_with_dot(self):
        result = extract_vendor_from_query("Total for IBM Corp.")
        assert result is not None
        assert "IBM" in result


class TestExtractDateRangeFromQuery:
    """Tests for extract_date_range_from_query()."""

    def test_full_year(self):
        date_from, date_to = extract_date_range_from_query("in 2024")
        assert date_from == "2024-01-01"
        assert date_to == "2024-12-31"

    def test_q1_2024(self):
        date_from, date_to = extract_date_range_from_query("Q1 2024")
        assert date_from == "2024-01-01"
        assert date_to == "2024-03-31"

    def test_q2_2024(self):
        date_from, date_to = extract_date_range_from_query("Q2 2024")
        assert date_from == "2024-04-01"
        assert date_to == "2024-06-30"

    def test_specific_date(self):
        date_from, date_to = extract_date_range_from_query("2024-07-20")
        assert date_from == "2024-07-20"
        assert date_to == "2024-07-20"

    def test_year_month(self):
        date_from, date_to = extract_date_range_from_query("2024-03")
        assert date_from == "2024-03-01"
        assert date_to == "2024-03-31"

    def test_month_name(self):
        date_from, date_to = extract_date_range_from_query("January 2024")
        assert date_from == "2024-01-01"
        assert date_to == "2024-01-31"

    def test_between_dates(self):
        date_from, date_to = extract_date_range_from_query(
            "between 2024-01-01 and 2024-06-30"
        )
        assert date_from == "2024-01-01"
        assert date_to == "2024-06-30"

    def test_no_date(self):
        date_from, date_to = extract_date_range_from_query("What is the total?")
        assert date_from is None
        assert date_to is None

    def test_last_month(self):
        date_from, date_to = extract_date_range_from_query("last month")
        assert date_from is not None
        assert date_to is not None

    def test_this_month(self):
        date_from, date_to = extract_date_range_from_query("this month")
        assert date_from is not None
        assert date_to is not None

    def test_last_year(self):
        date_from, date_to = extract_date_range_from_query("last year")
        assert date_from is not None
        assert date_to is not None