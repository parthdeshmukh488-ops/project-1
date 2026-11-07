"""Implied volatility: round trips, bounds, conditioning and honesty."""

import numpy as np
import pytest

from volsurf.black import CALL, PUT, forward_price, price
from volsurf.implied import _atm_exact, implied_vol, implied_vol_normalized
from volsurf.normal import norm_cdf

EPS = float(np.finfo(float).eps)


def _otm_grid(market):
    """A wide grid of out-of-the-money quotes -- the market convention."""
    spot, rate, carry = market["spot"], market["rate"], market["carry"]
    strike = np.exp(np.linspace(-6, 6, 401)) * spot
    expiry = np.array([1 / 365, 0.02, 0.1, 0.5, 1.0, 5.0, 20.0])[:, None]
    vol = np.array([0.005, 0.05, 0.2, 0.6, 1.5, 4.0])[:, None, None]
    fwd = forward_price(spot, rate, carry, expiry)
    option_type = np.where(strike > fwd, CALL, PUT)
    return strike, expiry, vol, option_type


def test_round_trip_reaches_the_conditioning_floor(market):
    """The central claim: recovered volatility is at the precision limit.

    ``condition`` is ``b / (s db/ds)``, the factor by which the problem
    amplifies a relative error in the input price.  Asserting against it rather
    than a flat tolerance is the only meaningful test: a quote one ulp below
    its no-arbitrage ceiling carries almost no information about volatility, so
    no solver can do better, while near the money the floor is 1 and a loose
    tolerance would hide genuine failure.
    """
    strike, expiry, vol, option_type = _otm_grid(market)
    quoted = price(market["spot"], strike, expiry, vol, market["rate"],
                   market["carry"], option_type)
    recovered, diag = implied_vol(
        quoted, market["spot"], strike, expiry, market["rate"], market["carry"],
        option_type, return_diagnostics=True,
    )

    ok = np.isfinite(recovered)
    assert ok.sum() > 10_000
    truth = np.broadcast_to(vol, recovered.shape)
    relative = np.abs(recovered[ok] / truth[ok] - 1.0)
    floor = np.maximum(diag.condition[ok], 1.0) * EPS

    assert np.median(relative) == 0.0, "most quotes should round-trip bit-exactly"
    assert np.max(relative / floor) < 10.0
    # Where the problem is well conditioned there is nowhere to hide.
    well_conditioned = diag.condition[ok] < 10.0
    assert np.max(relative[well_conditioned]) < 1e-14


def test_converges_in_a_handful_of_iterations(market):
    strike, expiry, vol, option_type = _otm_grid(market)
    quoted = price(market["spot"], strike, expiry, vol, market["rate"],
                   market["carry"], option_type)
    _, diag = implied_vol(
        quoted, market["spot"], strike, expiry, market["rate"], market["carry"],
        option_type, return_diagnostics=True,
    )
    ok = diag.converged & np.isfinite(diag.total_vol)
    assert diag.iterations[ok].mean() < 6.0


def test_at_the_money_closed_form_is_exact():
    """``b(0, s) = 2 Phi(s/2) - 1`` inverts in one inverse-normal call."""
    s = np.linspace(0.01, 5.0, 200)
    beta = 2.0 * norm_cdf(0.5 * s) - 1.0
    assert np.allclose(_atm_exact(beta), s, rtol=1e-13, atol=0.0)


def test_root_exactly_on_a_bracket_endpoint_is_found():
    """A regression: the bracket is closed, and the endpoint can be the answer.

    When the total volatility coincides with the inflection point
    ``s* = sqrt(2|x|)``, or with a doubling step of the upper bracket, the root
    sits exactly on an endpoint.  Treating the bracket as open rejected that
    value and bisected away from it -- while the zero residual simultaneously
    reported convergence.
    """
    x = -0.002
    s_star = np.sqrt(2.0 * abs(x))
    from volsurf.black import normalized_black

    for s_true in (s_star, 1.0, 2.0, 4.0):
        beta = normalized_black(x, s_true)
        got = implied_vol_normalized(np.array([x]), np.asarray(beta).reshape(1))
        assert float(got.total_vol[0]) == pytest.approx(s_true, rel=1e-12)


def test_prices_outside_no_arbitrage_bounds_return_nan(market):
    spot, rate, carry = market["spot"], market["rate"], market["carry"]
    strike, expiry = 110.0, 1.0
    fwd = float(forward_price(spot, rate, carry, expiry))
    discount = np.exp(-rate * expiry)

    below_intrinsic = discount * max(fwd - strike, 0.0) - 1.0
    above_ceiling = discount * fwd * 1.01
    out = implied_vol(
        np.array([below_intrinsic, above_ceiling, -1.0]),
        spot, strike, expiry, rate, carry, CALL,
    )
    assert np.all(np.isnan(out))


def test_subnormal_prices_are_rejected_rather_than_guessed():
    """Below the smallest normal double a price has under 53 bits of mantissa.

    Answering such a quote would mean reporting a volatility the input does not
    determine, so the solver declines instead.
    """
    tiny = np.finfo(float).tiny
    out = implied_vol_normalized(np.array([-0.5]), np.array([tiny / 4.0]))
    assert np.isnan(out.total_vol[0])
    assert not out.converged[0]


def test_deep_in_the_money_quotes_report_that_they_cannot_be_inverted():
    """The honest case: the time value is below double precision.

    A one-week call six volatilities in the money has a time value around
    ``1e-100`` against a price of order one.  Recovering it means subtracting
    the intrinsic, which leaves nothing at all -- the subtraction returns
    exactly the intrinsic, so the reported significance is zero ulps and no
    setting of the floor can produce an answer.  Returning a confident wrong
    number here is the easiest way for a library to look better than it is.
    """
    spot, strike, expiry, vol = 100.0, 2.0, 1 / 365, 0.005
    quoted = price(spot, strike, expiry, vol, 0.0, 0.0, CALL)
    recovered, diag = implied_vol(
        quoted, spot, strike, expiry, 0.0, 0.0, CALL, return_diagnostics=True
    )
    assert np.isnan(recovered)
    assert not bool(diag.converged)
    assert float(diag.time_value_ulps) == 0.0

    # Not a tolerance that can be relaxed: there is no information to recover.
    forced = implied_vol(
        quoted, spot, strike, expiry, 0.0, 0.0, CALL, min_time_value_ulps=0.0
    )
    assert np.isnan(forced)


@pytest.mark.parametrize(
    "expiry,vol,strike",
    [
        (0.10, 0.12, 80.0),
        (0.05, 0.08, 88.0),
        (0.05, 0.10, 85.0),
        (0.03, 0.10, 88.0),
        (0.02, 0.15, 85.0),
    ],
)
def test_reported_significance_predicts_the_attainable_accuracy(expiry, vol, strike):
    """``time_value_ulps`` is a real forecast, not a label.

    These quotes span five orders of magnitude of significance, from 250,000
    ulps down to under one.  With the floor disabled the solver returns
    something for all of them, and the relative error tracks ``1 / ulps``
    across the whole range -- which is what makes the diagnostic worth
    reporting and the default floor worth having.
    """
    spot = 100.0
    quoted = price(spot, strike, expiry, vol, 0.0, 0.0, CALL)
    recovered, diag = implied_vol(
        quoted, spot, strike, expiry, 0.0, 0.0, CALL,
        min_time_value_ulps=0.0, return_diagnostics=True,
    )
    ulps = float(diag.time_value_ulps)
    assert 0.0 < ulps < 1e6
    error = abs(float(recovered) / vol - 1.0)
    assert error < 5.0 / ulps, f"{ulps:.3g} ulps gave {error:.2e} relative error"

    # And the default floor refuses every one of them.
    assert np.isnan(implied_vol(quoted, spot, strike, expiry, 0.0, 0.0, CALL))


def test_in_the_money_quotes_invert_when_the_time_value_survives(market):
    """Parity is used, so a well-resolved in-the-money quote is still fine."""
    spot, rate, carry = market["spot"], market["rate"], market["carry"]
    strike = np.array([80.0, 90.0, 100.0, 110.0, 125.0])
    expiry, vol = 1.0, 0.25
    quoted = price(spot, strike, expiry, vol, rate, carry, CALL)
    recovered = implied_vol(quoted, spot, strike, expiry, rate, carry, CALL)
    assert np.allclose(recovered, vol, rtol=1e-10, atol=0.0)


def test_puts_and_calls_agree_through_parity(market):
    spot, rate, carry = market["spot"], market["rate"], market["carry"]
    strike = np.linspace(60.0, 160.0, 41)
    expiry, vol = 0.9, 0.3
    from_calls = implied_vol(
        price(spot, strike, expiry, vol, rate, carry, CALL),
        spot, strike, expiry, rate, carry, CALL,
    )
    from_puts = implied_vol(
        price(spot, strike, expiry, vol, rate, carry, PUT),
        spot, strike, expiry, rate, carry, PUT,
    )
    assert np.allclose(from_calls, from_puts, rtol=1e-9, atol=0.0)


def test_diagnostics_broadcast_to_the_output_shape(market):
    strike = np.array([90.0, 110.0])[:, None]
    expiry = np.array([0.5, 1.0, 2.0])[None, :]
    quoted = price(market["spot"], strike, expiry, 0.3, 0.02, 0.01, CALL)
    out, diag = implied_vol(
        quoted, market["spot"], strike, expiry, 0.02, 0.01, CALL,
        return_diagnostics=True,
    )
    assert out.shape == (2, 3)
    for field in (diag.iterations, diag.converged, diag.time_value_ulps,
                  diag.condition):
        assert field.shape == (2, 3)
