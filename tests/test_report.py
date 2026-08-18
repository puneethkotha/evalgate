"""Tests for the dashboard report payload."""

from evalgate.report import build_dashboard_report


def test_report_payload_shape():
    r = build_dashboard_report(seed=0)
    assert set(r) >= {"date", "agent", "run", "min_kappa", "gate", "taxonomy", "history"}
    assert r["gate"]["passed"] is True
    assert r["gate"]["calibration"]["kappa"] > 0.7
    assert len(r["taxonomy"]) >= 1


def test_report_history_has_drift_episode():
    r = build_dashboard_report(seed=0, min_kappa=0.70)
    drifted = [h for h in r["history"] if h["drifted"]]
    assert drifted, "expected at least one drifted run in the timeline"
    # Every drifted run's kappa is genuinely below the threshold.
    assert all(h["kappa"] < 0.70 for h in drifted)
    # The final (current) run matches the live gate.
    assert r["history"][-1]["run"] == r["run"]


def test_report_is_json_serializable():
    import json

    r = build_dashboard_report(seed=0)
    json.dumps(r)  # must not raise
