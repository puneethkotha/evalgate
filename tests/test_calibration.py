"""Tests for the judge-calibration crux."""

import math

from evalgate.calibration import calibrate


def test_perfect_agreement_kappa_one_not_drifted():
    # Mixed labels (both classes present) that agree perfectly => kappa == 1.
    human = [True, False, True, False, True, False]
    judge = [True, False, True, False, True, False]

    report = calibrate(judge, human, min_kappa=0.7)

    assert math.isclose(report.kappa, 1.0, abs_tol=1e-9)
    assert report.tpr == 1.0
    assert report.tnr == 1.0
    assert report.n == 6
    assert report.drifted is False


def test_strong_disagreement_flags_drift():
    # Judge inverts the humans on most anchors => kappa well below threshold => drift.
    human = [True, True, True, False, False, False]
    judge = [False, False, True, True, True, False]

    report = calibrate(judge, human, min_kappa=0.7)

    assert report.kappa < 0.7
    assert report.drifted is True


def test_kappa_corrects_for_chance_below_threshold():
    # A judge that agrees ~50% by luck on balanced classes has kappa ~ 0 -> drifted.
    human = [True, False, True, False, True, False, True, False]
    judge = [True, True, False, False, True, True, False, False]

    report = calibrate(judge, human, min_kappa=0.7)

    assert report.kappa < 0.7
    assert report.drifted is True


def test_degenerate_single_class_fails_closed():
    # Judge marks everything pass; kappa is undefined (NaN) => treated as drifted.
    human = [True, False, True, False]
    judge = [True, True, True, True]

    report = calibrate(judge, human, min_kappa=0.7)

    assert math.isnan(report.kappa)
    assert report.drifted is True


def test_length_mismatch_raises():
    try:
        calibrate([True, False], [True], min_kappa=0.7)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched label lengths")
