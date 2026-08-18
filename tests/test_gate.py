"""Tests for the structured CI gate."""

from evalgate.calibration import calibrate
from evalgate.gate import evaluate_gate, render_gate


def test_gate_passes_on_strong_sample():
    report = evaluate_gate([True] * 100, min_pass_rate=0.90)
    assert report.passed is True
    assert report.pass_rate.lower >= 0.90


def test_gate_fails_on_ci_lower_bound():
    # 9/10 looks like 0.90 but the Wilson lower bound is far below it.
    report = evaluate_gate([True] * 9 + [False], min_pass_rate=0.90)
    assert report.passed is False
    assert any("lower bound" in r for r in report.reasons)
    assert report.pass_rate.lower < 0.90


def test_gate_fails_on_judge_drift():
    human = [True, True, True, False, False, False]
    drifted_judge = [False, False, False, True, True, True]  # inverted -> kappa < 0
    calib = calibrate(drifted_judge, human, min_kappa=0.7)
    assert calib.drifted is True
    report = evaluate_gate([True] * 100, min_pass_rate=0.90, calibration=calib)
    assert report.passed is False
    assert any("drift" in r for r in report.reasons)


def test_gate_fails_on_significant_regression():
    baseline = [True] * 20
    current = [False] * 12 + [True] * 8  # 12 things that used to pass now fail
    report = evaluate_gate(current, min_pass_rate=0.0, baseline_passes=baseline)
    assert report.delta is not None
    assert report.delta.verdict == "regressed"
    assert report.passed is False
    assert any("regression" in r for r in report.reasons)


def test_gate_reports_improvement_without_failing():
    baseline = [False] * 8 + [True] * 12
    current = [True] * 20
    report = evaluate_gate(current, min_pass_rate=0.0, baseline_passes=baseline)
    assert report.delta.verdict == "improved"
    assert report.passed is True


def test_bias_corrected_only_when_judge_derived():
    human = [True, True, True, False, False, False]
    judge = [True, True, False, False, False, True]  # imperfect but agreeing enough
    calib = calibrate(judge, human, min_kappa=0.0)
    passes = [True] * 45 + [False] * 5
    code_report = evaluate_gate(passes, 0.0, calibration=calib, judge_derived=False)
    judge_report = evaluate_gate(passes, 0.0, calibration=calib, judge_derived=True)
    assert code_report.pass_rate.corrected is None
    assert judge_report.pass_rate.corrected is not None


def test_empty_sample_fails():
    report = evaluate_gate([], min_pass_rate=0.90)
    assert report.passed is False


def test_render_gate_borders_align():
    report = evaluate_gate([True] * 47 + [False] * 3, min_pass_rate=0.90)
    lines = render_gate(report).splitlines()
    widths = {len(line) for line in lines}
    assert len(widths) == 1  # every line (incl. borders) is the same width
