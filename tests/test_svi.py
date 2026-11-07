"""SVI: analytic derivatives, the density, and a calibration that is a function."""

import numpy as np
import pytest
from scipy.integrate import quad

from volsurf.black import normalized_black
from volsurf.svi import (
    SVIParams,
    butterfly_g,
    calibrate_slice,
    check_calendar_spread,
    d2_total_variance_dk2,
    d_total_variance_dk,
    risk_neutral_density,
    total_variance,
)

SLICE = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.2, expiry=1.0)


def test_parameter_validation():
    with pytest.raises(ValueError):
        SVIParams(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        SVIParams(a=0.04, b=0.4, rho=1.5, m=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        SVIParams(a=0.04, b=0.4, rho=0.0, m=0.0, sigma=0.0)


def test_analytic_first_derivative():
    k = np.linspace(-1.2, 1.2, 25)
    step = 1e-6
    fd = (total_variance(SLICE, k + step) - total_variance(SLICE, k - step)) / (2 * step)
    assert np.allclose(fd, d_total_variance_dk(SLICE, k), rtol=1e-8, atol=0.0)


def test_analytic_second_derivative():
    k = np.linspace(-1.2, 1.2, 25)
    step = 1e-4
    fd = (
        total_variance(SLICE, k + step)
        - 2 * total_variance(SLICE, k)
        + total_variance(SLICE, k - step)
    ) / step**2
    assert np.allclose(fd, d2_total_variance_dk2(SLICE, k), rtol=1e-5, atol=0.0)


def test_second_derivative_is_strictly_positive():
    """SVI is convex in k by construction, which is why it cannot oscillate."""
    k = np.linspace(-6.0, 6.0, 1001)
    assert np.all(d2_total_variance_dk2(SLICE, k) > 0.0)


def test_minimum_variance_matches_a_numerical_minimum():
    k = np.linspace(-3.0, 3.0, 200_001)
    assert float(total_variance(SLICE, k).min()) == pytest.approx(
        SLICE.minimum_variance, rel=1e-8
    )


def test_density_is_a_probability_density():
    """Total mass one and a correct forward: both must hold, and do.

    The integration range matters: SVI wings are linear in ``k``, so the
    density decays exponentially rather than Gaussian-fast and truncating at
    three standard deviations loses measurable mass.
    """
    mass = quad(lambda k: float(risk_neutral_density(SLICE, k)), -40, 40, limit=800)[0]
    forward = quad(
        lambda k: float(np.exp(k) * risk_neutral_density(SLICE, k)), -40, 40, limit=800
    )[0]
    assert mass == pytest.approx(1.0, abs=1e-9)
    assert forward == pytest.approx(1.0, abs=1e-9)


def test_density_matches_the_curvature_of_actual_option_prices():
    """Ties :mod:`volsurf.svi` to :mod:`volsurf.black`.

    The Gatheral-Jacquier density is derived from the implied-variance
    parameterisation; the same quantity is the second derivative of the call
    price with respect to strike.  Agreement means the parameterisation, its
    analytic derivatives and the pricer all tell the same story.  With
    ``c(k)`` the undiscounted call on a unit forward at strike ``exp(k)``,
    ``p(k) = exp(-k) (c'' - c')``.
    """

    def call(k):
        s = np.sqrt(float(total_variance(SLICE, k)))
        return float(normalized_black(-k, s)) * np.exp(0.5 * k)

    step = 1e-4
    for k in (-0.8, -0.5, -0.1, 0.0, 0.2, 0.6, 1.0):
        second = (call(k + step) - 2 * call(k) + call(k - step)) / step**2
        first = (call(k + step) - call(k - step)) / (2 * step)
        numeric = np.exp(-k) * (second - first)
        assert numeric == pytest.approx(
            float(risk_neutral_density(SLICE, k)), rel=1e-6
        )


def test_calibration_recovers_planted_parameters():
    k = np.linspace(-0.6, 0.6, 15)
    fit = calibrate_slice(k, total_variance(SLICE, k), expiry=1.0)
    assert np.allclose(fit.as_array(), SLICE.as_array(), rtol=1e-5, atol=1e-7)


def test_calibration_is_deterministic():
    """A fit that feeds a risk number must be a function of its inputs.

    This is the reason for the two-dimensional reduction: with three of the
    five parameters solved exactly, the remaining search is small enough to
    multistart exhaustively, so the same quotes always give the same answer.
    A direct five-parameter optimisation does not have this property.
    """
    k = np.linspace(-0.5, 0.5, 13)
    w = total_variance(SLICE, k) * (1 + 1e-4 * np.sin(17 * k))
    first = calibrate_slice(k, w, expiry=1.0).as_array()
    for _ in range(3):
        assert np.array_equal(calibrate_slice(k, w, expiry=1.0).as_array(), first)


def test_calibration_respects_the_parameter_constraints():
    """The reduction's bounds are the no-arbitrage constraints, so they hold.

    Noisy quotes are exactly what pushes an unconstrained fit outside
    ``|rho| <= 1`` and ``b >= 0``; here those are box bounds on the exactly
    solved inner problem and cannot be violated.
    """
    rng = np.random.default_rng(7)
    k = np.linspace(-0.7, 0.7, 19)
    for _ in range(12):
        noisy = total_variance(SLICE, k) * (1 + rng.normal(0, 0.03, k.size))
        fit = calibrate_slice(k, np.maximum(noisy, 1e-6), expiry=1.0)
        assert fit.b >= 0.0
        assert -1.0 <= fit.rho <= 1.0
        assert fit.sigma > 0.0


def test_calibration_of_a_flat_smile_does_not_produce_an_invalid_slice():
    """A regression: no skew means no correlation, not a NaN.

    With a flat smile the inner solve drives ``c`` to zero and ``rho = d / c``
    came back as NaN, which then failed the ``SVIParams`` bounds check several
    calls later -- during a surface build, where the cause is far from obvious.
    """
    k = np.linspace(-0.3, 0.3, 11)
    flat = np.full(k.size, 0.04)
    fit = calibrate_slice(k, flat, expiry=0.01)
    assert np.isfinite(fit.as_array()).all()
    assert np.allclose(total_variance(fit, k), flat, atol=1e-8)


def test_calibration_needs_enough_quotes():
    with pytest.raises(ValueError):
        calibrate_slice(np.linspace(-0.2, 0.2, 4), np.full(4, 0.04))
    with pytest.raises(ValueError):
        calibrate_slice(np.linspace(-0.2, 0.2, 7), np.full(5, 0.04))


def test_calendar_spread_check_detects_a_decreasing_surface():
    rising = [
        SVIParams(a=0.02 * t, b=0.3, rho=-0.3, m=0.0, sigma=0.15, expiry=t)
        for t in (0.25, 0.5, 1.0)
    ]
    assert check_calendar_spread(rising) > 0.0

    falling = [
        SVIParams(a=0.05, b=0.3, rho=-0.3, m=0.0, sigma=0.15, expiry=0.25),
        SVIParams(a=0.01, b=0.3, rho=-0.3, m=0.0, sigma=0.15, expiry=1.0),
    ]
    assert check_calendar_spread(falling) < 0.0


def test_butterfly_function_flags_an_arbitrageable_slice():
    """A large ``b`` with a narrow centre is the classic violation."""
    bad = SVIParams(a=0.001, b=1.2, rho=-0.9, m=0.0, sigma=0.02, expiry=1.0)
    k = np.linspace(-0.5, 0.5, 2001)
    assert float(butterfly_g(bad, k).min()) < 0.0
    assert float(butterfly_g(SLICE, k).min()) > 0.0
