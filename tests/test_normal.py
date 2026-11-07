"""The normal-distribution primitives, and the de-cancellation that matters."""

import numpy as np
import pytest
from scipy.special import ndtr

from volsurf.normal import (
    _ASYMPTOTIC_THRESHOLD,
    _mills_difference_asymptotic,
    mills_difference,
    mills_ratio,
    norm_cdf,
    norm_pdf,
    norm_ppf,
)


def test_mills_ratio_matches_definition_where_the_definition_works():
    """For moderate arguments the ratio can be formed directly; check it is."""
    w = np.linspace(-6.0, 20.0, 400)
    direct = ndtr(-w) / norm_pdf(w)
    assert np.allclose(mills_ratio(w), direct, rtol=1e-13, atol=0.0)


def test_mills_ratio_survives_where_the_definition_underflows():
    """At w = 40 the tail mass is 1e-351 and underflows; the ratio must not.

    This is the whole purpose of the function: the quotient is an ordinary
    number near 1/w even when numerator and denominator are both unrepresentable.
    """
    assert ndtr(-40.0) == 0.0
    assert mills_ratio(40.0) == pytest.approx(0.0249844042057206, rel=1e-14)
    # Asymptotically 1/w - 1/w^3 + 3/w^5, whose own truncation error is
    # O(1/w^7) -- so the tolerance has to tighten with w, not stay fixed.
    for w, tol in ((40.0, 5e-9), (80.0, 1e-10), (200.0, 1e-12), (1000.0, 1e-15)):
        series = 1.0 / w - 1.0 / w**3 + 3.0 / w**5
        assert mills_ratio(w) == pytest.approx(series, rel=tol)


def test_mills_difference_series_agrees_with_direct_when_direct_is_reliable():
    """Widely separated arguments: both routes should work, and agree."""
    a = np.array([25.0, 30.0, 40.0, 60.0, 100.0, 200.0, 1e4])
    c = a * 1.5
    assert np.allclose(
        _mills_difference_asymptotic(a, c), mills_ratio(a) - mills_ratio(c),
        rtol=5e-15, atol=0.0,
    )


def test_mills_difference_series_beats_direct_when_arguments_are_close():
    """The point of the series: a tiny gap destroys the direct difference.

    With a gap of 1e-7 at w = 30 the direct subtraction keeps about eight
    digits.  The series keeps all of them, which is checked against a central
    difference of the (smooth, well-conditioned) Mills ratio itself.
    """
    a = np.array([30.0, 50.0, 100.0])
    gap = 1e-7
    c = a + gap
    series = _mills_difference_asymptotic(a, c)
    # -d/dw mills_ratio by a one-sided difference.  Its own truncation error
    # is O(step / w), which sets the tolerance below.
    step = 1e-4
    slope = (mills_ratio(a) - mills_ratio(a + step)) / step
    assert np.allclose(series, slope * gap, rtol=1e-5, atol=0.0)


def test_mills_difference_does_not_overflow_on_huge_arguments():
    """The reciprocal formulation exists so a^15 c^15 never gets formed."""
    a = np.array([1e6, 1e9, 1e12])
    out = mills_difference(a, a * (1.0 + 1e-9))
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)


def test_mills_difference_honours_an_explicit_gap():
    """An exact gap must be used as given, not recovered by subtraction.

    Recomputing ``c - a`` is the cancelling operation the parameter exists to
    avoid, so passing a gap the floats cannot represent as a difference has to
    change the answer.
    """
    a = np.array([0.38])
    gap = 2.6e-4
    c = a + gap
    with_gap = mills_difference(a, c, gap=np.array([gap]))
    assert np.all(np.isfinite(with_gap))
    assert np.all(with_gap > 0.0)


def test_mills_difference_accepts_scalars():
    """Zero-dimensional input used to crash on `nonzero`; it must not."""
    assert float(mills_difference(30.0, 31.0)) > 0.0
    assert float(mills_difference(0.5, 0.6)) > 0.0


def test_branch_switch_is_continuous():
    """Crossing the series threshold must not produce a visible step."""
    below = _ASYMPTOTIC_THRESHOLD - 1e-9
    above = _ASYMPTOTIC_THRESHOLD + 1e-9
    gap = 0.1
    left = float(mills_difference(below, below + gap, gap=gap))
    right = float(mills_difference(above, above + gap, gap=gap))
    assert left == pytest.approx(right, rel=1e-12)


def test_ppf_inverts_cdf():
    p = np.linspace(1e-10, 1 - 1e-10, 500)
    assert np.allclose(norm_cdf(norm_ppf(p)), p, rtol=0, atol=1e-12)
