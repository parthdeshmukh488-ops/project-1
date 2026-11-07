"""The Black formula: known values, parity, Greeks, and the precision claim."""

import numpy as np
import pytest

from reference import (
    EPS,
    exact_conditioning,
    exact_normalized_black,
    requires_mpmath,
    textbook_normalized_black,
    ulp_error,
)
from volsurf.black import (
    CALL,
    PUT,
    forward_price,
    greeks,
    normalized_black,
    normalized_vega,
    price,
)


def test_textbook_reference_values():
    """Hull's worked example, to nine decimals."""
    assert float(price(100, 100, 1.0, 0.2, 0.05, 0.0, "call")) == pytest.approx(
        10.450583572185, rel=1e-12
    )
    assert float(price(100, 100, 1.0, 0.2, 0.05, 0.0, "put")) == pytest.approx(
        5.573526022256, rel=1e-12
    )


def test_put_call_parity_holds_to_machine_precision():
    rng = np.random.default_rng(0)
    spot = rng.uniform(20, 200, 300)
    strike = rng.uniform(20, 200, 300)
    expiry = rng.uniform(0.01, 10.0, 300)
    vol = rng.uniform(0.02, 2.0, 300)
    rate = rng.uniform(-0.01, 0.1, 300)
    carry = rng.uniform(0.0, 0.08, 300)

    call = price(spot, strike, expiry, vol, rate, carry, CALL)
    put = price(spot, strike, expiry, vol, rate, carry, PUT)
    forward = spot * np.exp(-carry * expiry) - strike * np.exp(-rate * expiry)
    assert np.allclose(call - put, forward, rtol=0, atol=1e-11)


def test_zero_volatility_and_zero_expiry_give_intrinsic():
    assert float(price(110, 100, 1.0, 0.0, 0.0, 0.0, CALL)) == pytest.approx(10.0)
    assert float(price(90, 100, 1.0, 0.0, 0.0, 0.0, CALL)) == pytest.approx(0.0)
    assert float(price(110, 100, 0.0, 0.3, 0.0, 0.0, CALL)) == pytest.approx(10.0)
    assert float(price(90, 100, 0.0, 0.3, 0.0, 0.0, PUT)) == pytest.approx(10.0)


def test_prices_respect_no_arbitrage_bounds():
    rng = np.random.default_rng(3)
    spot, strike = 100.0, rng.uniform(10, 400, 500)
    expiry, vol, rate, carry = 1.3, rng.uniform(0.01, 3.0, 500), 0.03, 0.01
    call = price(spot, strike, expiry, vol, rate, carry, CALL)
    fwd = float(forward_price(spot, rate, carry, expiry))
    discount = np.exp(-rate * expiry)
    assert np.all(call >= discount * np.maximum(fwd - strike, 0.0) - 1e-12)
    assert np.all(call <= discount * fwd + 1e-12)


def test_monotone_in_volatility():
    vol = np.linspace(0.01, 3.0, 500)
    call = price(100, 110, 0.7, vol, 0.03, 0.01, CALL)
    assert np.all(np.diff(call) > 0)


def test_broadcasting_over_every_argument():
    spot = np.array([90.0, 100.0, 110.0])[:, None, None]
    strike = np.array([95.0, 105.0])[None, :, None]
    expiry = np.array([0.5, 1.0, 2.0, 3.0])[None, None, :]
    out = price(spot, strike, expiry, 0.25, 0.02, 0.01, CALL)
    assert out.shape == (3, 2, 4)
    assert float(out[1, 0, 2]) == pytest.approx(
        float(price(100.0, 95.0, 2.0, 0.25, 0.02, 0.01, CALL))
    )


def test_options_on_futures_reduce_to_black_76():
    """Setting the carry equal to the rate makes the forward the spot."""
    f, k, t, v, r = 100.0, 95.0, 0.8, 0.3, 0.04
    got = float(price(f, k, t, v, r, r, CALL))
    d1 = (np.log(f / k) + 0.5 * v * v * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    from scipy.special import ndtr

    expected = np.exp(-r * t) * (f * ndtr(d1) - k * ndtr(d2))
    assert got == pytest.approx(expected, rel=1e-13)


def test_normalized_vega_is_the_derivative_of_the_normalized_price():
    x = np.array([-2.0, -0.5, -0.05, 0.0, 0.3])
    s = np.array([0.1, 0.3, 0.8, 0.2, 1.5])
    step = 1e-7
    fd = (normalized_black(x, s + step) - normalized_black(x, s - step)) / (2 * step)
    assert np.allclose(fd, normalized_vega(x, s), rtol=1e-7, atol=0.0)


@pytest.mark.parametrize(
    "spot,strike,expiry,vol,rate,carry",
    [
        (100.0, 95.0, 0.8, 0.25, 0.04, 0.015),
        (100.0, 120.0, 0.25, 0.40, 0.00, 0.000),
        (50.0, 45.0, 2.0, 0.15, 0.05, 0.070),
        (100.0, 100.0, 0.05, 0.60, 0.02, 0.000),
    ],
)
@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_every_greek_against_finite_differences(
    spot, strike, expiry, vol, rate, carry, option_type
):
    """Each sensitivity differenced numerically out of :func:`price`.

    This is the only test that catches a wrong coefficient in a Greek that is
    otherwise plausible: ``charm`` was initially written with ``q`` where the
    derivative of ``d1`` requires the net carry ``r - q``, an error of
    ``S gamma (r - 2q)`` that no reference-value check would have flagged.
    """

    def evaluate(**overrides):
        args = dict(
            spot=spot, strike=strike, expiry=expiry, vol=vol, rate=rate, carry=carry
        )
        args.update(overrides)
        return float(price(option_type=option_type, **args))

    def first(fn, at, step):
        return (fn(at + step) - fn(at - step)) / (2 * step)

    def second(fn, at, step):
        return (fn(at + step) - 2 * fn(at) + fn(at - step)) / step**2

    g = greeks(spot, strike, expiry, vol, rate, carry, option_type)
    h_s, h_v = 1e-3, 1e-5

    checks = {
        "delta": (g.delta, first(lambda s: evaluate(spot=s), spot, h_s), 1e-6),
        "gamma": (g.gamma, second(lambda s: evaluate(spot=s), spot, h_s), 1e-4),
        "vega": (g.vega, first(lambda v: evaluate(vol=v), vol, h_v), 1e-7),
        "theta": (g.theta, -first(lambda t: evaluate(expiry=t), expiry, 1e-5), 1e-7),
        "rho": (g.rho, first(lambda r: evaluate(rate=r), rate, 1e-6), 1e-7),
        "dual_delta": (
            g.dual_delta,
            first(lambda k: evaluate(strike=k), strike, 1e-3),
            1e-7,
        ),
        "vanna": (
            g.vanna,
            first(
                lambda v: first(lambda s: evaluate(spot=s, vol=v), spot, h_s), vol, h_v
            ),
            1e-5,
        ),
        "volga": (g.volga, second(lambda v: evaluate(vol=v), vol, 1e-4), 1e-5),
        "charm": (
            g.charm,
            -first(
                lambda t: first(lambda s: evaluate(spot=s, expiry=t), spot, h_s),
                expiry,
                1e-4,
            ),
            1e-4,
        ),
        "speed": (
            g.speed,
            first(lambda s: second(lambda z: evaluate(spot=z), s, 1e-2), spot, 1e-2),
            1e-3,
        ),
        "zomma": (
            g.zomma,
            first(
                lambda v: second(lambda s: evaluate(spot=s, vol=v), spot, 1e-2),
                vol,
                1e-4,
            ),
            1e-3,
        ),
    }
    for name, (analytic, numeric, tol) in checks.items():
        assert float(analytic) == pytest.approx(numeric, rel=tol), name


def test_greeks_price_matches_price_function():
    g = greeks(100, 95, 0.8, 0.25, 0.04, 0.015, CALL)
    assert float(g.price) == pytest.approx(
        float(price(100, 95, 0.8, 0.25, 0.04, 0.015, CALL)), rel=1e-15
    )


def test_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        price(100, 100, 1.0, 0.2, option_type="straddle")
    with pytest.raises(ValueError):
        price(100, 100, 1.0, 0.2, option_type=2)


# --- the precision claim ------------------------------------------------


@requires_mpmath
def test_normalized_black_reaches_its_conditioning_floor(wing_grid):
    """The central accuracy claim, measured rather than asserted loosely.

    A flat tolerance would be the wrong test: deep out of the money the price
    is genuinely sensitive to an ulp of its own inputs, so no routine can do
    better than ``kappa`` ulps, while near the money ``kappa`` is 1 and a loose
    bound would hide real error.  The claim is therefore relative to the floor.
    """
    x, s = wing_grid
    ratios, errors = [], []
    for xi, si in zip(x, s):
        exact = exact_normalized_black(xi, si)
        if exact == 0 or exact < 1e-290:
            continue
        err = ulp_error(normalized_black(xi, si), exact)
        floor = max(exact_conditioning(xi, si), 1.0)
        errors.append(err)
        ratios.append(err / floor)

    assert len(ratios) > 1000
    assert np.median(errors) < 3.0, f"median {np.median(errors):.2f} ulps"
    assert max(ratios) < 10.0, f"worst case {max(ratios):.1f}x the floor"


@requires_mpmath
def test_beats_the_textbook_formula_by_orders_of_magnitude(wing_grid):
    """The rearrangement has to earn its complexity; this is by how much."""
    x, s = wing_grid
    ours, theirs = [], []
    for xi, si in zip(x, s):
        exact = exact_normalized_black(xi, si)
        if exact == 0 or exact < 1e-290:
            continue
        naive = textbook_normalized_black(xi, si)
        if naive <= 0:
            continue
        ours.append(ulp_error(normalized_black(xi, si), exact))
        theirs.append(ulp_error(naive, exact))

    assert np.median(theirs) > 5 * np.median(ours)
    # The worst case is where it really matters: the textbook form loses
    # everything in the short-dated wings.
    assert max(theirs) > 1e6 * max(ours) / 1e3
    assert max(theirs) > 1e6


@requires_mpmath
def test_accurate_at_prices_far_below_any_tolerance():
    """A normalized price of 2.4e-141 is still right to a few ulps."""
    exact = exact_normalized_black(-0.5, 0.02)
    assert float(exact) < 1e-140
    assert ulp_error(normalized_black(-0.5, 0.02), exact) < 50.0
    # ... whereas the textbook formula has lost eight digits by then.
    assert ulp_error(textbook_normalized_black(-0.5, 0.02), exact) > 1e4
