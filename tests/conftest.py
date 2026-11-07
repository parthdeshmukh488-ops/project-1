"""Fixtures shared across the suite."""

import numpy as np
import pytest


@pytest.fixture
def wing_grid():
    """Random ``(x, s)`` pairs spanning the whole admissible domain.

    Log-uniform in both, so the short-dated wings -- where every formula here
    is hard -- carry as much weight as the comfortable at-the-money region
    that a linear grid would be dominated by.
    """
    rng = np.random.default_rng(1234)
    n = 1500
    x = -np.exp(rng.uniform(np.log(1e-6), np.log(8.0), n))
    s = np.exp(rng.uniform(np.log(1e-4), np.log(4.0), n))
    return x, s


@pytest.fixture
def market():
    """A plain vanilla market: spot, rate and dividend yield."""
    return dict(spot=100.0, rate=0.025, carry=0.012)
