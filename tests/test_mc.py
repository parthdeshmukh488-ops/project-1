"""Monte Carlo under a local volatility surface: exactness, controls, bias."""

import numpy as np
import pytest

from volsurf.black import forward_price
from volsurf.black import price as black_price
from volsurf.mc import convergence_study, price_european, simulate_local_vol
from volsurf.surface import LocalVolSurface
from volsurf.svi import SVIParams

MARKET = dict(rate=0.025, carry=0.012)
SPOT = 100.0


def flat_surface(vol=0.2):
    return LocalVolSurface(
        [
            SVIParams(a=vol**2 * t, b=0.0, rho=0.0, m=0.0, sigma=0.3, expiry=t)
            for t in (0.1, 0.25, 0.5, 1.0, 2.0)
        ]
    )


def skewed_surface():
    return LocalVolSurface(
        [
            SVIParams(a=0.038 * t, b=0.09 * t, rho=-0.75, m=0.03, sigma=0.12,
                      expiry=t)
            for t in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
        ]
    )


@pytest.mark.parametrize("log_moneyness", [-0.15, 0.0, 0.15])
@pytest.mark.parametrize("expiry", [0.5, 1.0])
def test_flat_surface_reproduces_black_without_a_control(log_moneyness, expiry):
    """The unit test for the simulation itself.

    On a flat surface the local volatility *is* the Black volatility, so the
    raw estimator -- no control variate, nothing to hide behind -- has to land
    on the closed form within its sampling error.  Any bug in the drift, the
    time stepping or the surface lookup shows up here.
    """
    surface = flat_surface(0.2)
    fwd = float(forward_price(SPOT, MARKET["rate"], MARKET["carry"], expiry))
    strike = fwd * np.exp(log_moneyness)
    option_type = 1 if log_moneyness >= 0 else -1

    result = price_european(
        surface, SPOT, strike, expiry, option_type=option_type,
        n_paths=200_000, n_steps=200, seed=3, use_control=False, **MARKET
    )
    exact = float(
        black_price(SPOT, strike, expiry, 0.2, MARKET["rate"], MARKET["carry"],
                    option_type)
    )
    assert abs(result.price - exact) < 4.0 * result.standard_error


def test_control_variate_is_exact_on_a_flat_surface():
    """When the control is the answer, the estimator should simply be it.

    On a flat surface the control variate's paths are the simulation's paths,
    so the regression removes the entire sampling error and returns the
    analytic price.  This is the degenerate case, and it is worth pinning
    because it confirms the control's mean is wired up correctly -- a wrong
    reference volatility would show as a bias here and nowhere else.
    """
    surface = flat_surface(0.2)
    expiry = 1.0
    fwd = float(forward_price(SPOT, MARKET["rate"], MARKET["carry"], expiry))
    result = price_european(
        surface, SPOT, fwd, expiry, n_paths=50_000, n_steps=100, seed=3, **MARKET
    )
    exact = float(
        black_price(SPOT, fwd, expiry, 0.2, MARKET["rate"], MARKET["carry"], 1)
    )
    assert result.price == pytest.approx(exact, abs=1e-9)
    assert result.standard_error < 1e-9


def test_control_variate_reduces_variance_on_a_real_surface():
    surface = skewed_surface()
    expiry = 1.0
    fwd = float(forward_price(SPOT, MARKET["rate"], MARKET["carry"], expiry))
    result = price_european(
        surface, SPOT, fwd, expiry, n_paths=100_000, n_steps=200, seed=11, **MARKET
    )
    assert result.variance_reduction > 5.0
    assert result.standard_error < result.raw_standard_error


def test_control_variate_does_not_bias_the_estimate():
    """Unbiasedness is the property that makes the control safe to use.

    The control's mean is known in closed form rather than simulated, so
    subtracting ``beta (control - mean)`` cannot move the expectation whatever
    ``beta`` is.  Checked by agreement with the uncontrolled estimator, which
    has no control to be wrong.
    """
    surface = skewed_surface()
    expiry = 0.5
    fwd = float(forward_price(SPOT, MARKET["rate"], MARKET["carry"], expiry))
    strike = fwd * 1.05
    controlled = price_european(
        surface, SPOT, strike, expiry, n_paths=200_000, n_steps=200, seed=4,
        **MARKET
    )
    raw = price_european(
        surface, SPOT, strike, expiry, n_paths=200_000, n_steps=200, seed=4,
        use_control=False, **MARKET
    )
    spread = np.hypot(controlled.standard_error, raw.standard_error)
    assert abs(controlled.price - raw.price) < 4.0 * spread


def test_simulation_is_a_martingale_in_the_forward():
    surface = skewed_surface()
    expiry = 1.0
    terminal, gbm, fallbacks = simulate_local_vol(
        surface, SPOT, expiry, n_paths=200_000, n_steps=400, seed=5, **MARKET
    )
    fwd = float(forward_price(SPOT, MARKET["rate"], MARKET["carry"], expiry))
    for sample in (terminal, gbm):
        standard_error = sample.std(ddof=1) / np.sqrt(sample.size)
        assert abs(sample.mean() - fwd) < 4.0 * standard_error
    assert fallbacks == 0, "a well-posed surface should never fail to give a vol"


def test_confidence_interval_brackets_the_estimate():
    surface = skewed_surface()
    result = price_european(
        surface, SPOT, 100.0, 0.5, n_paths=20_000, n_steps=100, seed=1, **MARKET
    )
    low, high = result.confidence_interval
    assert low < result.price < high
    assert "paths" in str(result)


@pytest.mark.slow
def test_euler_bias_shrinks_with_the_step_size():
    """Variance reduction does nothing about discretisation error.

    Without this the residual gap on a real surface is ambiguous -- an
    ``O(dt)`` bias looks exactly like a model error.  Refining the grid has to
    move the answer monotonically towards the analytic price, and on a flat
    surface that price is known exactly, so the bias can be read off directly
    rather than inferred.
    """
    surface = flat_surface(0.35)
    expiry = 1.0
    fwd = float(forward_price(SPOT, MARKET["rate"], MARKET["carry"], expiry))
    strike = fwd * 0.9
    exact = float(
        black_price(SPOT, strike, expiry, 0.35, MARKET["rate"], MARKET["carry"], 1)
    )
    rows = convergence_study(
        surface, SPOT, strike, expiry, (10, 40, 160), use_control=False,
        n_paths=400_000, seed=9, **MARKET
    )
    errors = [abs(result.price - exact) for _, result in rows]
    assert errors[-1] < errors[0]
