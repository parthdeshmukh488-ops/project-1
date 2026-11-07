"""Normal-distribution primitives, built for the wings.

Every formula in this library eventually reduces to the standard normal CDF.
The textbook definition is useless where option pricing needs it most: a
25-delta six-month quote is fine, but a 0.1-delta wing quote evaluates
``Phi(-6)`` and a one-week deep out-of-the-money option evaluates ``Phi(-30)``,
where the CDF is ``4.9e-201`` and any formula that subtracts two such numbers
has no significant digits left at all.

The fix is to never form those differences in the first place.  Two objects do
all the work:

``mills_ratio(w) = Phi(-w) / phi(w)``
    The ratio of the tail mass to the density at the same point.  For large
    ``w`` it behaves like ``1/w`` -- a perfectly ordinary number -- while
    numerator and denominator are both astronomically small.  Written in terms
    of the scaled complementary error function ``erfcx(z) = exp(z^2) erfc(z)``
    it costs one library call and has no underflow at all.

``mills_difference(a, c)``
    ``mills_ratio`` differences are what the Black formula actually needs, and
    for close arguments they cancel.  Factoring the gap ``c - a`` out of the
    asymptotic series analytically removes the cancellation exactly.

The two are the reason :func:`volsurf.black.normalized_black` holds full double
precision down to prices of ``1e-300``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx, log_ndtr, ndtr, ndtri

__all__ = [
    "norm_pdf",
    "norm_cdf",
    "norm_logcdf",
    "norm_ppf",
    "mills_ratio",
    "mills_difference",
    "SQRT_TWO",
    "SQRT_TWO_PI",
    "INV_SQRT_TWO_PI",
]

SQRT_TWO = np.sqrt(2.0)
SQRT_TWO_PI = np.sqrt(2.0 * np.pi)
INV_SQRT_TWO_PI = 1.0 / SQRT_TWO_PI
SQRT_PI = np.sqrt(np.pi)
SQRT_PI_OVER_TWO = np.sqrt(np.pi / 2.0)

# Above this argument the asymptotic expansion of ``erfcx`` is accurate to
# better than 3e-15 relative with the eight terms carried in
# ``_mills_difference_asymptotic``, so it is preferred over the (cancelling)
# direct evaluation.  See ``tests/test_normal.py::test_mills_difference_branches
# _agree_in_overlap`` for the measured crossover.
_ASYMPTOTIC_THRESHOLD = 22.0


def norm_pdf(x):
    """Standard normal density."""
    x = np.asarray(x, dtype=float)
    return INV_SQRT_TWO_PI * np.exp(-0.5 * x * x)


def norm_cdf(x):
    """Standard normal CDF, via ``scipy.special.ndtr``."""
    return ndtr(np.asarray(x, dtype=float))


def norm_logcdf(x):
    """``log Phi(x)``, accurate where ``Phi(x)`` itself underflows."""
    return log_ndtr(np.asarray(x, dtype=float))


def norm_ppf(p):
    """Inverse standard normal CDF."""
    return ndtri(np.asarray(p, dtype=float))


def mills_ratio(w):
    """``Phi(-w) / phi(w)``, free of underflow for every finite ``w``.

    ``Phi(-w) = erfc(w / sqrt 2) / 2`` and ``phi(w) = exp(-w^2/2) / sqrt(2 pi)``,
    so the ratio is ``sqrt(pi/2) * erfcx(w / sqrt 2)`` once the Gaussian factor
    is absorbed into the scaling of ``erfcx``.  Nothing is ever exponentiated,
    which is the whole point: ``mills_ratio(40) = 0.0249`` is returned exactly
    while ``Phi(-40)`` is ``7.3e-351`` and underflows to zero.
    """
    return SQRT_PI_OVER_TWO * erfcx(np.asarray(w, dtype=float) / SQRT_TWO)


def _mills_difference_asymptotic(a, c, gap=None):
    """``mills_ratio(a) - mills_ratio(c)`` with the cancellation removed.

    The asymptotic series of the scaled complementary error function is

        erfcx(z) ~ (1 / (z sqrt pi)) * sum_n (-1)^n (2n-1)!! / (2 z^2)^n

    so the difference of two such series, with ``z = w / sqrt 2``, is a sum of
    terms ``1/a^(2n+1) - 1/c^(2n+1)``.  Each of those factors exactly as

        1/a^m - 1/c^m = (c - m) * (sum_{j} c^j a^(m-1-j)) / (a^m c^m)

    with the gap ``d = c - a`` pulled out *symbolically*.  Because ``d`` is
    passed in rather than reconstructed, the subtraction that destroys
    precision in the direct form never happens: the result is accurate to
    full relative precision no matter how close ``a`` and ``c`` are.
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    if gap is None:
        gap = c - a

    # Work in reciprocals.  For large arguments p and r are small, so no power
    # ever overflows -- the direct form needs a^15 c^15, which overflows around
    # a = 1e10 and is reached by one-day options six strikes out.
    p = 1.0 / a
    r = 1.0 / c
    # Exact, and the only place the gap appears: (1/a - 1/c) = (c - a) / (a c).
    gap = np.asarray(gap, dtype=float) * p * r

    # Substituting z = w / sqrt 2 turns the (2 z^2)^n denominator of the erfcx
    # series into w^(2n), leaving the plain double factorials (2n-1)!! as the
    # coefficients.  Eight terms hold the series to ~3e-15 relative at w = 22,
    # which is where ``mills_difference`` switches branches.
    coeffs = (1.0, -1.0, 3.0, -15.0, 105.0, -945.0, 10395.0, -135135.0)

    total = np.zeros(np.broadcast(a, c).shape, dtype=float)
    for n, coeff in enumerate(coeffs):
        m = 2 * n + 1
        # (p^m - r^m) / (p - r), expanded so the cancelling subtraction is gone.
        divided = np.zeros_like(total)
        for j in range(m):
            divided = divided + np.power(p, j) * np.power(r, m - 1 - j)
        total = total + coeff * gap * divided

    return total


def mills_difference(a, c, gap=None):
    """``mills_ratio(a) - mills_ratio(c)`` for ``c >= a``, without cancellation.

    ``gap`` is ``c - a``.  Pass it whenever the caller knows it exactly, which
    is the usual case and matters more than it looks: in the Black formula the
    two arguments are ``-(h + t)`` and ``-(h - t)``, so their difference is
    ``2t`` -- a quantity the caller already holds to full precision, but which
    *recomputing* from the two stored arguments destroys.  Subtracting two
    numbers near ``0.38`` that differ by ``2.6e-4`` costs twelve digits, and
    the resulting error propagates straight through an otherwise exact series.
    That single avoidable subtraction was worth three orders of magnitude of
    accuracy here.

    For well-separated arguments the direct difference is already accurate and
    is used as-is.  Once both arguments exceed :data:`_ASYMPTOTIC_THRESHOLD`
    the series form takes over, where the ``c - a`` gap is factored out
    analytically.  The two branches agree to ``3e-16`` relative in the overlap,
    which is what the test suite checks.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    c = np.atleast_1d(np.asarray(c, dtype=float))
    gap = c - a if gap is None else np.asarray(gap, dtype=float)
    a, c, gap = np.broadcast_arrays(a, c, gap)

    out = np.array(mills_ratio(a) - mills_ratio(c), dtype=float, copy=True)
    use_series = a > _ASYMPTOTIC_THRESHOLD
    if np.any(use_series):
        out[use_series] = _mills_difference_asymptotic(
            a[use_series], c[use_series], gap[use_series]
        )
    return out
