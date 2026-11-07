"""Heston: the branch cut, the martingale identity, and three routes agreeing."""

import numpy as np
import pytest

from volsurf.black import forward_price
from volsurf.black import price as black_price
from volsurf.heston import (
    HestonParams,
    characteristic_function,
    price,
    price_fft,
    simulate,
)

EQUITY = HestonParams(v0=0.045, kappa=1.3, theta=0.05, xi=0.65, rho=-0.72)
MARKET = dict(spot=100.0, rate=0.025, carry=0.012)


def test_parameter_validation():
    with pytest.raises(ValueError):
        HestonParams(v0=-0.1, kappa=1.0, theta=0.04, xi=0.5, rho=0.0)
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=0.0, theta=0.04, xi=0.5, rho=0.0)
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-1.5)


def test_feller_condition_is_reported():
    assert EQUITY.feller == pytest.approx(2 * 1.3 * 0.05 / 0.65**2)
    assert EQUITY.feller < 1.0  # calibrated equity parameters usually do


@pytest.mark.parametrize("expiry", [0.01, 0.1, 1.0, 5.0, 10.0, 30.0])
def test_characteristic_function_identities_hold_at_every_maturity(expiry):
    """``phi(0) = 1`` normalises, ``phi(-i) = 1`` is the martingale condition.

    These two are the sharpest cheap test of the whole expression, and the
    reason the maturity list runs to thirty years: the textbook arrangement of
    the Heston characteristic function crosses the principal branch cut of the
    complex logarithm at long maturities, and ``phi(-i)`` stops being one well
    before the oscillation becomes visible in prices.
    """
    assert characteristic_function(EQUITY, 0.0, expiry) == pytest.approx(1.0, abs=1e-14)
    martingale = characteristic_function(EQUITY, -1j, expiry)
    assert complex(martingale).real == pytest.approx(1.0, abs=1e-11)
    assert complex(martingale).imag == pytest.approx(0.0, abs=1e-11)


def test_characteristic_function_is_hermitian():
    """``phi(-u) = conj(phi(u))`` because the underlying variable is real."""
    u = np.linspace(0.1, 50.0, 200)
    assert np.allclose(
        characteristic_function(EQUITY, -u, 1.5),
        np.conj(characteristic_function(EQUITY, u, 1.5)),
        rtol=1e-12,
        atol=0.0,
    )


def test_characteristic_function_modulus_decays():
    """Needed for the Fourier integrals to converge at all."""
    u = np.array([1.0, 10.0, 50.0, 100.0, 200.0, 400.0])
    modulus = np.abs(characteristic_function(EQUITY, u, 1.0))
    assert np.all(np.diff(modulus) < 0)
    # Measured: 9.7e-11 by u = 200 and 6.1e-21 by u = 400, which is why the
    # adaptive cutoff in `price` settles well inside a few hundred.
    assert modulus[-1] < 1e-20


def test_put_call_parity():
    for strike in (70.0, 100.0, 140.0):
        for expiry in (0.25, 1.0, 3.0):
            call = price(EQUITY, strike=strike, expiry=expiry, option_type="call",
                         **MARKET)
            put = price(EQUITY, strike=strike, expiry=expiry, option_type="put",
                        **MARKET)
            fwd = MARKET["spot"] * np.exp(-MARKET["carry"] * expiry)
            bond = strike * np.exp(-MARKET["rate"] * expiry)
            assert call - put == pytest.approx(fwd - bond, abs=1e-10)


def test_degenerates_to_black_at_the_right_order_in_vol_of_vol():
    """With ``v0 = theta`` and ``xi -> 0`` the variance freezes.

    Asserting the *order* of convergence rather than a tolerance at one ``xi``,
    because the order is the substantive claim and a fixed tolerance would pass
    for the wrong reasons.  A correlated vol-of-vol generates skew at
    ``O(xi)``, so the gap to Black is first order; switching the correlation
    off removes that term and leaves ``O(xi^2)``.  Both are checked, and the
    second is what would catch a sign error in the ``rho`` terms.
    """
    reference = float(
        black_price(MARKET["spot"], 100.0, 1.0, 0.2, MARKET["rate"],
                    MARKET["carry"], 1)
    )

    def gap(xi, rho):
        frozen = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=xi, rho=rho)
        return abs(price(frozen, strike=100.0, expiry=1.0, **MARKET) - reference)

    # First order with skew: the gap falls by ten for every decade of xi.
    for xi in (1e-2, 1e-3, 1e-4):
        assert gap(xi, -0.7) == pytest.approx(0.153 * xi, rel=0.25)

    # Second order without it: by a hundred.
    ratio = gap(1e-2, 0.0) / gap(1e-3, 0.0)
    assert 70.0 < ratio < 130.0


def test_prices_respect_no_arbitrage_bounds():
    expiry = 1.0
    fwd = float(forward_price(MARKET["spot"], MARKET["rate"], MARKET["carry"], expiry))
    discount = np.exp(-MARKET["rate"] * expiry)
    for strike in (40.0, 80.0, 100.0, 150.0, 250.0):
        call = price(EQUITY, strike=strike, expiry=expiry, **MARKET)
        assert call >= discount * max(fwd - strike, 0.0) - 1e-10
        assert call <= discount * fwd + 1e-10


@pytest.mark.parametrize("expiry", [0.1, 0.5, 1.0, 3.0, 10.0])
def test_fft_agrees_with_the_quadrature_reference(expiry):
    """Two Fourier routes, different algebra, same answer.

    The bound is absolute rather than relative on purpose: for the extreme
    strikes on the FFT's own grid the option is worth less than ``1e-13``, and
    a relative tolerance there would be a statement about the option, not the
    method.
    """
    strikes, calls = price_fft(EQUITY, MARKET["spot"], expiry, MARKET["rate"],
                               MARKET["carry"])
    liquid = (strikes > 55.0) & (strikes < 180.0)
    sampled = np.linspace(0, int(liquid.sum()) - 1, 10).astype(int)
    chosen = strikes[liquid][sampled]
    reference = np.array(
        [price(EQUITY, strike=float(k), expiry=expiry, **MARKET) for k in chosen]
    )
    assert np.allclose(calls[liquid][sampled], reference, rtol=0, atol=1e-11)


@pytest.mark.slow
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_monte_carlo_agrees_with_the_characteristic_function(strike):
    """The genuinely independent check.

    The quadrature and the FFT are two rearrangements of the same algebra; if
    the characteristic function were wrong they would agree with each other and
    both be wrong.  A Monte Carlo of the stochastic differential equations
    shares none of it, so agreement to within the sampling error is evidence
    about the formula rather than about the rearrangement.
    """
    expiry = 1.0
    terminal = simulate(
        EQUITY, MARKET["spot"], expiry, n_paths=400_000, n_steps=600,
        rate=MARKET["rate"], carry=MARKET["carry"], seed=1,
    )
    payoff = np.maximum(terminal - strike, 0.0) * np.exp(-MARKET["rate"] * expiry)
    standard_error = payoff.std(ddof=1) / np.sqrt(payoff.size)
    exact = price(EQUITY, strike=strike, expiry=expiry, **MARKET)
    assert abs(payoff.mean() - exact) < 4.0 * standard_error


@pytest.mark.slow
def test_simulation_is_a_martingale_in_the_forward():
    expiry = 1.0
    terminal = simulate(
        EQUITY, MARKET["spot"], expiry, n_paths=200_000, n_steps=400,
        rate=MARKET["rate"], carry=MARKET["carry"], seed=5,
    )
    fwd = float(forward_price(MARKET["spot"], MARKET["rate"], MARKET["carry"], expiry))
    standard_error = terminal.std(ddof=1) / np.sqrt(terminal.size)
    assert abs(terminal.mean() - fwd) < 4.0 * standard_error


@pytest.mark.slow
def test_full_truncation_keeps_paths_finite_when_feller_fails():
    """Variance will go negative; the scheme must cope rather than produce NaN."""
    violent = HestonParams(v0=0.01, kappa=0.5, theta=0.02, xi=1.5, rho=-0.8)
    assert violent.feller < 0.1
    terminal = simulate(violent, 100.0, 2.0, n_paths=20_000, n_steps=400, seed=2)
    assert np.all(np.isfinite(terminal))
    assert np.all(terminal > 0.0)
