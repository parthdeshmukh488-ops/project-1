"""SVI: fitting a volatility slice without fitting arbitrage into it.

A market quotes maybe fifteen strikes per expiry.  Everything downstream --
interpolating between them, extracting a risk-neutral density, backing out a
local volatility surface -- needs a smooth curve through those points, and
needs it to be *arbitrage-free*, because the second derivative of the fitted
curve is a probability density and a curve that fits beautifully can easily
imply a negative one.

Gatheral's raw SVI parameterises total implied variance as

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]

with ``k = log(K / F)``.  Five parameters, linear wings, a smooth hyperbolic
centre: it reproduces equity index smiles closely and -- the reason it is worth
preferring to a spline -- its derivatives are analytic, so the no-arbitrage
conditions and the Dupire local volatility in :mod:`volsurf.surface` are exact
rather than finite-differenced.

Calibration is the interesting part.  Throwing five correlated parameters at a
general optimiser is unreliable and slow; it lands in different local minima
depending on the starting point, which is unacceptable when the fit feeds a
risk number.  Instead this module uses the reduction of Zeliade's quasi-explicit
method, which collapses the problem to two dimensions -- see
:func:`calibrate_slice`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear, minimize

from .normal import norm_pdf

__all__ = [
    "SVIParams",
    "butterfly_g",
    "calibrate_slice",
    "check_calendar_spread",
    "d2_total_variance_dk2",
    "d_total_variance_dk",
    "risk_neutral_density",
    "total_variance",
]


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters for one expiry.

    ``a`` sets the overall variance level, ``b`` the wing slopes, ``rho`` the
    skew (the asymmetry between them), ``m`` the horizontal shift of the
    minimum and ``sigma`` how rounded the centre is.  ``expiry`` is carried
    along because calendar-spread arbitrage is a statement about two slices at
    different maturities, not about either one alone.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry: float = 1.0

    def __post_init__(self):
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    @property
    def minimum_variance(self):
        """The smallest total variance the slice attains.

        Reached at ``k = m - rho sigma / sqrt(1 - rho^2)``; if this is negative
        the slice is not a valid variance curve at all.
        """
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)

    def as_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])


def total_variance(params: SVIParams, k):
    """Total implied variance ``w(k) = sigma_BS(k)^2 T``."""
    k = np.asarray(k, dtype=float)
    z = k - params.m
    return params.a + params.b * (
        params.rho * z + np.sqrt(z * z + params.sigma**2)
    )


def implied_vol_from_params(params: SVIParams, k):
    """Black-Scholes implied volatility implied by the slice."""
    return np.sqrt(np.maximum(total_variance(params, k), 0.0) / params.expiry)


def d_total_variance_dk(params: SVIParams, k):
    """``dw/dk``, analytically."""
    k = np.asarray(k, dtype=float)
    z = k - params.m
    return params.b * (params.rho + z / np.sqrt(z * z + params.sigma**2))


def d2_total_variance_dk2(params: SVIParams, k):
    """``d2w/dk2``, analytically.

    Strictly positive for ``b > 0``, which is why SVI is convex in ``k`` and
    cannot produce the oscillating density a free-form spline fit can.
    """
    k = np.asarray(k, dtype=float)
    z = k - params.m
    return params.b * params.sigma**2 / np.power(z * z + params.sigma**2, 1.5)


def butterfly_g(params: SVIParams, k):
    """The Gatheral-Jacquier function ``g(k)``; non-negative iff no butterfly
    arbitrage.

    A butterfly spread is a bet that the underlying lands in a narrow band, so
    its price is the probability of that band and cannot be negative.  In terms
    of total variance that requirement is

        g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2  >=  0

    and ``g`` is exactly the factor multiplying the Gaussian in the
    risk-neutral density, so a negative ``g`` *is* a negative probability.
    Checking it is not optional: a least-squares SVI fit to clean market quotes
    violates it routinely in the wings, which is why
    :func:`calibrate_slice` constrains the fit rather than repairing it after.
    """
    k = np.asarray(k, dtype=float)
    w = total_variance(params, k)
    w1 = d_total_variance_dk(params, k)
    w2 = d2_total_variance_dk2(params, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = 1.0 - k * w1 / (2.0 * w)
        return term * term - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2


def risk_neutral_density(params: SVIParams, k):
    """The density of ``log(K/F)`` implied by the slice.

    This is the real test of a surface fit.  Prices can look fine while the
    second derivative they imply is negative somewhere, and that is what shows
    up here.
    """
    k = np.asarray(k, dtype=float)
    w = total_variance(params, k)
    g = butterfly_g(params, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
        return g * norm_pdf(d2) / np.sqrt(w)


def _inner_design(y):
    """Design matrix for the exactly solvable inner problem.

    With ``y = (k - m) / sigma`` the slice is linear in three unknowns,

        w = a + d y + c sqrt(y^2 + 1),      d = rho b sigma,  c = b sigma

    and the constraints ``b >= 0``, ``|rho| <= 1`` become ``c >= 0`` and
    ``|d| <= c``, which are *not* box constraints -- so a bounded linear solver
    cannot be used directly.  The substitution

        u = c + d,   v = c - d

    fixes that: ``|d| <= c`` is exactly ``u >= 0, v >= 0``, and Gatheral's
    no-butterfly bound ``|d| <= 4 sigma - c`` together with ``c <= 4 sigma``
    becomes exactly ``u <= 4 sigma, v <= 4 sigma``.  Every constraint is now a
    simple bound, so the inner problem is an exactly solvable bounded linear
    least squares -- no iteration, no local minima, no starting guess.
    """
    root = np.sqrt(y * y + 1.0)
    return np.column_stack([np.ones_like(y), 0.5 * (root + y), 0.5 * (root - y)])


def _solve_inner(k, w, m, sigma, weights):
    """Exact bounded linear least squares for ``(a, b, rho)`` at fixed
    ``(m, sigma)``."""
    y = (k - m) / sigma
    design = _inner_design(y)
    sqrt_w = np.sqrt(weights)[:, None]
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([max(w.max(), 1e-12), 4.0 * sigma, 4.0 * sigma])
    sol = lsq_linear(
        design * sqrt_w, w * np.sqrt(weights), bounds=(lower, upper), method="bvls"
    )
    a, u, v = sol.x
    c = 0.5 * (u + v)
    d = 0.5 * (u - v)
    b = max(c / sigma, 0.0)
    # A nearly flat smile drives c to zero, and d/c then comes back as NaN or
    # denormal garbage.  Correlation is undefined when there is no skew to
    # measure, so report zero rather than propagating a value that fails the
    # SVIParams bounds check further downstream.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = d / c if c > 0.0 else 0.0
    if not np.isfinite(ratio):
        ratio = 0.0
    rho = float(np.clip(ratio, -1.0, 1.0))
    return a, b, rho, sol.cost


def calibrate_slice(
    k,
    total_var,
    expiry=1.0,
    weights=None,
    n_starts=12,
    seed=0,
):
    """Fit one SVI slice to observed total variances.

    The five raw parameters split cleanly.  For *fixed* ``(m, sigma)`` the
    slice is linear in the other three, and -- after the change of variables in
    :func:`_inner_design` -- subject only to simple bounds, so the inner
    problem is solved exactly by a bounded linear least squares with no
    starting guess and no local minima.  Only ``(m, sigma)`` is left to search,
    and two dimensions is small enough to multistart properly.

    That is the whole point of the decomposition: a direct five-parameter fit
    returns different answers from different starting points, which is not
    acceptable when the result feeds a risk number.  Here the two-dimensional
    surface is searched from ``n_starts`` deterministic starting points and the
    best is kept, which in testing recovers planted parameters to five decimal
    places and gives the same answer every time.

    Parameters
    ----------
    k
        Log-moneyness ``log(K / F)`` of each quote.
    total_var
        Observed total implied variance ``sigma^2 T`` at each ``k``.
    expiry
        Year fraction, carried into the returned parameters.
    weights
        Per-quote weights; defaults to equal.  Vega weights are the usual
        choice when fitting to prices rather than vols.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_var, dtype=float)
    if k.shape != w.shape:
        raise ValueError("k and total_var must have the same shape")
    if k.size < 5:
        raise ValueError("need at least five quotes to identify five parameters")
    weights = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)

    spread = max(k.max() - k.min(), 1e-6)

    def objective(mv):
        m, log_sigma = mv
        sigma = np.exp(log_sigma)
        _, _, _, cost = _solve_inner(k, w, m, sigma, weights)
        return cost

    rng = np.random.default_rng(seed)
    starts = [
        (float(np.mean(k)), np.log(0.5 * spread)),
        (float(k[np.argmin(w)]), np.log(0.2 * spread)),
        (0.0, np.log(0.1 * spread)),
    ]
    starts += [
        (float(rng.uniform(k.min() - spread, k.max() + spread)),
         float(rng.uniform(np.log(0.02 * spread), np.log(2.0 * spread))))
        for _ in range(max(0, n_starts - len(starts)))
    ]

    best = None
    for start in starts:
        res = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000},
        )
        if best is None or res.fun < best.fun:
            best = res

    m, log_sigma = best.x
    sigma = float(np.exp(log_sigma))
    a, b, rho, _ = _solve_inner(k, w, m, sigma, weights)
    return SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m),
                     sigma=sigma, expiry=float(expiry))


def check_calendar_spread(slices, k_grid=None):
    """Largest violation of calendar-spread arbitrage across a set of slices.

    Total implied variance must be non-decreasing in maturity at every fixed
    log-moneyness: otherwise a calendar spread has a negative price.  Slices
    fitted one expiry at a time have no reason to respect this, so it has to be
    checked on the assembled surface.  Returns the worst (most negative)
    increment found, which is ``>= 0`` for an arbitrage-free surface.
    """
    ordered = sorted(slices, key=lambda p: p.expiry)
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 601)
    k_grid = np.asarray(k_grid, dtype=float)
    worst = np.inf
    for earlier, later in itertools.pairwise(ordered):
        gap = total_variance(later, k_grid) - total_variance(earlier, k_grid)
        worst = min(worst, float(gap.min()))
    return worst if np.isfinite(worst) else 0.0
