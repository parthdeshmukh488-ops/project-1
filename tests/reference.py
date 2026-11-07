"""Independent references for the accuracy claims.

The accuracy work in :mod:`volsurf.black` only means something if it is
measured against something that did not come out of the same reasoning.  Two
references are used here, both deliberately naive:

* an **arbitrary-precision** evaluation of the Black formula straight from the
  definition, with no rearrangement at all -- at 50 digits none is needed, and
  that is exactly what makes it independent of the double-precision work;
* the **conditioning** of the price at each point, which bounds what any
  double-precision routine could achieve and is therefore the right yardstick.

``mpmath`` is an optional dependency; tests needing it are marked ``exact`` and
skip cleanly when it is absent.
"""

import numpy as np
import pytest

try:
    import mpmath as mp

    mp.mp.dps = 50
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover - only on a bare install
    HAVE_MPMATH = False

requires_mpmath = pytest.mark.skipif(
    not HAVE_MPMATH, reason="needs mpmath for the arbitrary-precision reference"
)

EPS = float(np.finfo(float).eps)


def _phi(z):
    return mp.erfc(-z / mp.sqrt(2)) / 2


def exact_normalized_black(x, s):
    """``b(x, s)`` to 50 significant digits, from the definition."""
    x = mp.mpf(float(x))
    s = mp.mpf(float(s))
    h, t = x / s, s / 2
    return mp.e ** (x / 2) * _phi(h + t) - mp.e ** (-x / 2) * _phi(h - t)


def exact_conditioning(x, s):
    """``|dlog b/dlog x| + |dlog b/dlog s|``: the error floor at ``(x, s)``.

    The density terms of ``db/dx`` cancel exactly, leaving
    ``db/dx = (A + B)/2`` with ``A`` and ``B`` the two terms of the formula --
    which is why the floor is computable in closed form at all.
    """
    xm = mp.mpf(float(x))
    sm = mp.mpf(float(s))
    h, t = xm / sm, sm / 2
    a = mp.e ** (xm / 2) * _phi(h + t)
    b = mp.e ** (-xm / 2) * _phi(h - t)
    price = a - b
    vega = mp.e ** (-(h**2 + t**2) / 2) / mp.sqrt(2 * mp.pi)
    return float(sm * vega / price + abs(xm) * (a + b) / (2 * price))


def ulp_error(computed, exact):
    """Relative error in units of machine epsilon."""
    if exact == 0:
        return 0.0
    return float(abs(mp.mpf(float(computed)) / exact - 1)) / EPS


def textbook_normalized_black(x, s):
    """The formula as written in the textbooks, for comparison.

    Kept here rather than in the package because its only purpose is to show
    how much is lost by evaluating the definition directly in double precision.
    """
    from scipy.special import ndtr

    h, t = x / s, s / 2
    return np.exp(0.5 * x) * ndtr(h + t) - np.exp(-0.5 * x) * ndtr(h - t)
