"""Tests for the extended calibration bundle (AC1, prevalence, band) + judge wiring."""

import math

from evalgate.calibration import calibrate, calibrate_judge, corrected_pass_rate
from evalgate.models import AnchorExample, JudgeResult


def test_bundle_fields_populated():
    human = [True, False, True, False, True, False, True, False]
    judge = [True, False, True, False, True, True, True, False]  # one FP
    r = calibrate(judge, human, min_kappa=0.0)
    assert 0.0 <= r.raw_agreement <= 1.0
    assert not math.isnan(r.ac1)
    assert r.prevalence == 0.5
    assert r.band in {"slight", "fair", "moderate", "substantial", "almost perfect"}
    assert math.isclose(r.fpr, 1 - r.tnr, abs_tol=1e-9)
    assert r.min_kappa == 0.0
    assert isinstance(r.paradox_flag, bool)


def test_perfect_judge_bundle():
    labels = [True, False, True, False]
    r = calibrate(labels, labels, min_kappa=0.7)
    assert math.isclose(r.kappa, 1.0)
    assert math.isclose(r.ac1, 1.0)
    assert r.band == "almost perfect"
    assert r.drifted is False


class _StubJudge:
    """Returns pass iff the output text contains 'good'."""

    def judge(self, input_text, output_text, rubric):
        return JudgeResult(passed=("good" in output_text), critique="stub")


def test_calibrate_judge_runs_judge_over_anchors():
    anchors = [
        AnchorExample(input="a", output="good plan", human_label=True),
        AnchorExample(input="b", output="bad plan", human_label=False),
        AnchorExample(input="c", output="good plan", human_label=True),
        AnchorExample(input="d", output="broken", human_label=False),
    ]
    r = calibrate_judge(_StubJudge(), anchors, rubric="r", min_kappa=0.7)
    assert r.n == 4
    assert math.isclose(r.kappa, 1.0)  # stub agrees with every human label
    assert r.drifted is False


def test_corrected_pass_rate():
    # Perfect judge (tpr=1, fpr=0) leaves the observed rate unchanged.
    assert corrected_pass_rate(0.9, tpr=1.0, fpr=0.0) == 0.9
    # Uninformative judge (tpr == fpr) -> undefined.
    assert corrected_pass_rate(0.9, tpr=0.5, fpr=0.5) is None
    # A judge that over-passes (high fpr) pulls the true rate below the observed.
    c = corrected_pass_rate(0.9, tpr=0.95, fpr=0.5)
    assert c is not None and c < 0.9
