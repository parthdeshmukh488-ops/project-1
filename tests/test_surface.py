"""The surface: the D == g identity, exact cases, and arbitrage detection."""

import numpy as np
import pytest

from volsurf.surface import LocalVolSurface, dupire_denominator
from volsurf.svi import (
    SVIParams,
    butterfly_g,
    d2_total_variance_dk2,
    d_total_variance_dk,
    total_variance,
)


def _flat_surface(vol=0.2, expiries=(0.1, 0.25, 0.5, 1.0, 2.0)):
    """Constant implied volatility: local volatility must equal it exactly."""
    return LocalVolSurface(
        [
            SVIParams(a=vol**2 * t, b=0.0, rho=0.0, m=0.0, sigma=0.3, expiry=t)
            for t in expiries
        ]
    )


def _skewed_surface():
    """A realistic equity surface: skewed, and flat in maturity.

    Scaling the whole slice with ``T`` -- ``a`` and ``b`` both proportional to
    it -- makes total variance ``T v(k)``, so implied volatility is the same
    smile at every expiry.  That is the "sticky implied volatility" surface,
    and it matters here that the term structure be built this way rather than
    by moving ``a`` alone: with only ``a`` growing, ``dw/dT`` has no strike
    dependence at all and the surface has essentially no *local* skew, which
    makes it useless for testing anything about skew.
    """
    return LocalVolSurface(
        [
            SVIParams(a=0.038 * t, b=0.09 * t, rho=-0.75, m=0.03, sigma=0.12,
                      expiry=t)
            for t in (0.1, 0.25, 0.5, 1.0, 2.0)
        ]
    )


def test_dupire_denominator_is_identically_the_butterfly_function():
    """The load-bearing identity of the whole module.

    Dupire's denominator comes from the forward PDE; the Gatheral-Jacquier
    ``g`` comes from requiring a butterfly spread to have a non-negative price.
    They are the same expression, which is *why* local volatility exists
    exactly when the surface is arbitrage-free.  The two are computed by
    separate functions precisely so this can be checked rather than assumed.
    """
    rng = np.random.default_rng(0)
    k = np.linspace(-3.0, 3.0, 401)
    for _ in range(25):
        params = SVIParams(
            a=float(rng.uniform(0.001, 0.2)),
            b=float(rng.uniform(0.01, 0.8)),
            rho=float(rng.uniform(-0.95, 0.95)),
            m=float(rng.uniform(-0.3, 0.3)),
            sigma=float(rng.uniform(0.05, 0.6)),
            expiry=float(rng.uniform(0.1, 3.0)),
        )
        denominator = dupire_denominator(
            total_variance(params, k),
            d_total_variance_dk(params, k),
            d2_total_variance_dk2(params, k),
            k,
        )
        assert np.allclose(denominator, butterfly_g(params, k), rtol=0, atol=1e-14)


def test_flat_surface_has_local_volatility_equal_to_implied():
    """With no skew and no term structure the two coincide exactly."""
    surface = _flat_surface(0.2)
    k = np.linspace(-1.0, 1.0, 21)
    for expiry in (0.15, 0.3, 0.75, 1.5):
        assert np.allclose(surface.local_vol(k, expiry), 0.2, rtol=1e-12, atol=0.0)
        assert np.allclose(surface.implied_vol(k, expiry), 0.2, rtol=1e-12, atol=0.0)


def test_piecewise_constant_term_structure_is_recovered_exactly():
    """A deterministic time-dependent volatility is an exact test case.

    For ``sigma(t)`` with no strike dependence the implied total variance is
    ``w(T) = int_0^T sigma^2``, so ``g == 1`` and Dupire reduces to
    ``sigma_loc^2(T) = dw/dT = sigma^2(T)``.  Putting the slices at the
    breakpoints makes the linear-in-T interpolation exact too, so the surface
    must return the instantaneous volatility to machine precision -- a complete
    end-to-end check of the extraction with no Monte Carlo noise in the way.
    """
    breaks = np.array([0.0, 0.25, 0.75, 1.5, 3.0])
    vols = np.array([0.30, 0.22, 0.18, 0.25])
    cumulative = np.concatenate([[0.0], np.cumsum(vols**2 * np.diff(breaks))])

    surface = LocalVolSurface(
        [
            SVIParams(a=float(w), b=0.0, rho=0.0, m=0.0, sigma=0.3, expiry=float(t))
            for t, w in zip(breaks[1:], cumulative[1:])
        ]
    )
    for i, vol in enumerate(vols[1:], start=1):
        midpoint = 0.5 * (breaks[i] + breaks[i + 1])
        assert float(surface.local_vol(0.0, midpoint)) == pytest.approx(vol, rel=1e-12)
        assert float(surface.local_vol(0.4, midpoint)) == pytest.approx(vol, rel=1e-12)


def test_local_vol_skew_exceeds_implied_vol_skew():
    """The standard rule of thumb, and a sanity check on the extraction.

    Local volatility skew is roughly twice implied volatility skew near the
    money, because implied volatility is an average of local volatility over
    the paths that reach a strike.  A sign error or a missing term in the
    Dupire denominator breaks this immediately.  Measured here: 2.14 at three
    months rising to 2.45 at a year, which is the expected direction too.
    """
    surface = _skewed_surface()
    step = 0.05
    for expiry, expected in ((0.25, 2.14), (0.5, 2.26), (1.0, 2.45)):
        local_slope = (
            float(surface.local_vol(step, expiry))
            - float(surface.local_vol(-step, expiry))
        ) / (2 * step)
        implied_slope = (
            float(surface.implied_vol(step, expiry))
            - float(surface.implied_vol(-step, expiry))
        ) / (2 * step)
        assert local_slope < 0 and implied_slope < 0
        assert local_slope / implied_slope == pytest.approx(expected, abs=0.05)


def test_arbitrage_free_surface_is_reported_as_such():
    diagnosis = _skewed_surface().diagnose()
    assert diagnosis.arbitrage_free
    assert diagnosis.min_butterfly > 0.0
    assert diagnosis.min_calendar > 0.0
    assert "arbitrage-free" in str(diagnosis)


def test_calendar_arbitrage_is_detected_and_makes_local_vol_undefined():
    """Total variance that falls with maturity is a negative numerator."""
    surface = LocalVolSurface(
        [
            SVIParams(a=0.06, b=0.2, rho=-0.2, m=0.0, sigma=0.2, expiry=0.5),
            SVIParams(a=0.01, b=0.2, rho=-0.2, m=0.0, sigma=0.2, expiry=1.0),
        ]
    )
    diagnosis = surface.diagnose()
    assert not diagnosis.arbitrage_free
    assert diagnosis.min_calendar < 0.0
    assert "ARBITRAGEABLE" in str(diagnosis)
    assert np.all(np.isnan(surface.local_vol(np.linspace(-0.3, 0.3, 11), 0.75)))


def test_butterfly_arbitrage_makes_local_vol_undefined_where_it_bites():
    """A negative denominator, reported as NaN rather than clipped.

    Flooring it at some small positive number would let a simulation run
    happily on prices that admit arbitrage, which is the one outcome worth
    refusing outright.
    """
    bad = [
        SVIParams(a=0.0005, b=1.4, rho=-0.92, m=0.0, sigma=0.015, expiry=t)
        for t in (0.5, 1.0)
    ]
    bad = [
        SVIParams(a=p.a * (1 + i), b=p.b, rho=p.rho, m=p.m, sigma=p.sigma,
                  expiry=p.expiry)
        for i, p in enumerate(bad)
    ]
    surface = LocalVolSurface(bad)
    diagnosis = surface.diagnose(k_grid=np.linspace(-0.3, 0.3, 1201))
    assert diagnosis.min_butterfly < 0.0
    assert not diagnosis.arbitrage_free
    k = np.linspace(-0.2, 0.2, 401)
    assert np.any(np.isnan(surface.local_vol(k, 0.75)))


def test_requires_at_least_two_expiries():
    with pytest.raises(ValueError):
        LocalVolSurface([SVIParams(a=0.04, b=0.3, rho=0.0, m=0.0, sigma=0.2, expiry=1.0)])
    with pytest.raises(ValueError):
        LocalVolSurface(
            [
                SVIParams(a=0.04, b=0.3, rho=0.0, m=0.0, sigma=0.2, expiry=1.0),
                SVIParams(a=0.05, b=0.3, rho=0.0, m=0.0, sigma=0.2, expiry=1.0),
            ]
        )


def test_spot_space_lookup_agrees_with_log_moneyness():
    surface = _skewed_surface()
    forward, expiry = 105.0, 0.8
    spot = np.array([80.0, 105.0, 130.0])
    assert np.allclose(
        surface.local_vol_from_spot(spot, forward, expiry),
        surface.local_vol(np.log(spot / forward), expiry),
        rtol=0, atol=0,
        equal_nan=True,
    )


def test_slices_may_be_supplied_in_any_order():
    ordered = _skewed_surface()
    shuffled = LocalVolSurface(list(reversed(ordered.slices)))
    k = np.linspace(-0.5, 0.5, 11)
    assert np.allclose(shuffled.local_vol(k, 0.7), ordered.local_vol(k, 0.7),
                       equal_nan=True)
