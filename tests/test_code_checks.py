"""Tests for the deterministic CodeChecks."""

from evalgate.evaluators import CodeChecks
from evalgate.models import Span, Trace

# --- schema_valid ---------------------------------------------------------------------

def test_schema_valid_accepts_matching_dict():
    schema = {"name": str, "age": int}
    assert CodeChecks.schema_valid({"name": "ada", "age": 36}, schema) is True


def test_schema_valid_accepts_json_string():
    schema = {"name": str}
    assert CodeChecks.schema_valid('{"name": "ada"}', schema) is True


def test_schema_valid_rejects_missing_key():
    schema = {"name": str, "age": int}
    assert CodeChecks.schema_valid({"name": "ada"}, schema) is False


def test_schema_valid_rejects_wrong_type():
    schema = {"age": int}
    assert CodeChecks.schema_valid({"age": "not-an-int"}, schema) is False


def test_schema_valid_rejects_bad_json():
    assert CodeChecks.schema_valid("{not json}", {"name": str}) is False


def test_schema_valid_none_type_only_requires_presence():
    assert CodeChecks.schema_valid({"anything": [1, 2, 3]}, {"anything": None}) is True


# --- no_pii ---------------------------------------------------------------------------

def test_no_pii_clean_text():
    assert CodeChecks.no_pii("the quick brown fox jumps over the lazy dog") is True


def test_no_pii_detects_email():
    assert CodeChecks.no_pii("reach me at ada@example.com") is False


def test_no_pii_detects_ssn():
    assert CodeChecks.no_pii("my ssn is 123-45-6789") is False


def test_no_pii_detects_phone():
    assert CodeChecks.no_pii("call +1 (415) 555-0123 tomorrow") is False


def test_no_pii_empty_string():
    assert CodeChecks.no_pii("") is True


# --- latency_budget -------------------------------------------------------------------

def _trace(*latencies: float) -> Trace:
    return Trace(
        trace_id="t",
        spans=[Span(name=f"s{i}", latency_ms=ms) for i, ms in enumerate(latencies)],
    )


def test_latency_budget_within_budget():
    assert CodeChecks.latency_budget(_trace(100.0, 200.0), ms=500.0) is True


def test_latency_budget_exceeds_budget():
    assert CodeChecks.latency_budget(_trace(300.0, 300.0), ms=500.0) is False


def test_latency_budget_boundary_is_inclusive():
    assert CodeChecks.latency_budget(_trace(250.0, 250.0), ms=500.0) is True
