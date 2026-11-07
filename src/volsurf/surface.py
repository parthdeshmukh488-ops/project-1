"""From quoted smiles to a local volatility surface.

A set of SVI slices is a surface in implied volatility: it says what the market
charges for every strike and expiry.  It is not yet a *model* -- it cannot be
simulated, and it cannot price a barrier or a cliquet.  Dupire's construction
turns it into one, by extracting the unique local volatility function
``sigma_loc(S, t)`` whose diffusion reproduces every quoted European price.

In terms of total implied variance ``w(k, T)`` with ``k = log(K / F_T)``,

    sigma_loc^2 = (dw/dT) / D,
    D = 1 - (k/w) w_k + (1/4)(-1/4 - 1/w + k^2/w^2) w_k^2 + (1/2) w_kk

and the thing worth noticing is that ``D`` is not a new object.  Expanding both
expressions shows ``D`` is *identically* the Gatheral-Jacquier butterfly
function ``g(k)`` of :mod:`volsurf.svi` -- the one whose non-negativity makes
the risk-neutral density a density.  So

    sigma_loc^2 = (dw/dT) / g(k)

and the two things that can go wrong are exactly the two classical arbitrages:
a negative denominator is butterfly arbitrage, a negative numerator is calendar
arbitrage.  **Local volatility exists precisely when the surface is
arbitrage-free**, and a complaint from this module is never a numerical problem
to be clipped away -- it is the fit telling you the surface it produced is not
a legitimate set of prices.  :func:`LocalVolSurface.diagnose` reports which of
the two failed and where.  The suite checks the identity ``D == g`` directly,
since it is the load-bearing claim.

Interpolation across expiry is linear in total variance at fixed
log-moneyness, which makes ``dw/dT`` piecewise constant and -- more
importantly -- makes it non-negative exactly when the slices satisfy the
calendar condition, so no interpolation artefact can manufacture an arbitrage
that the inputs did not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .svi import (
    SVIParams,
    butterfly_g,
    d2_total_variance_dk2,
    d_total_variance_dk,
    total_variance,
)

__all__ = ["LocalVolSurface", "SurfaceDiagnosis", "dupire_denominator"]


def dupire_denominator(w, w_k, w_kk, k):
    """``D``, the Dupire denominator, written out as Dupire's formula gives it.

    Kept separate from :func:`volsurf.svi.butterfly_g` on purpose: the two are
    algebraically the same quantity reached from completely different
    directions -- one from the forward PDE, one from the positivity of a
    butterfly spread -- and the test suite asserts they agree.  Collapsing them
    into one function would make that check vacuous.
    """
    w = np.asarray(w, dtype=float)
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            1.0
            - (k / w) * w_k
            + 0.25 * (-0.25 - 1.0 / w + k * k / (w * w)) * w_k * w_k
            + 0.5 * w_kk
        )


@dataclass(frozen=True)
class SurfaceDiagnosis:
    """Which arbitrage a surface commits, and how badly.

    ``min_butterfly`` is the smallest ``g(k, T)`` found and ``min_calendar``
    the smallest ``dw/dT``; both must be non-negative.  ``arbitrage_free``
    allows a small tolerance, because a fit that lands within rounding of zero
    is not committing an arbitrage, it is touching the boundary.
    """

    min_butterfly: float
    min_calendar: float
    worst_butterfly_k: float
    worst_calendar_k: float
    tolerance: float

    @property
    def arbitrage_free(self):
        return (
            self.min_butterfly >= -self.tolerance
            and self.min_calendar >= -self.tolerance
        )

    def __str__(self):
        verdict = "arbitrage-free" if self.arbitrage_free else "ARBITRAGEABLE"
        return (
            f"surface is {verdict}: min butterfly g = {self.min_butterfly:+.3e} "
            f"at k = {self.worst_butterfly_k:+.3f}, "
            f"min dw/dT = {self.min_calendar:+.3e} "
            f"at k = {self.worst_calendar_k:+.3f}"
        )


class LocalVolSurface:
    """A volatility surface assembled from SVI slices, with Dupire local vol.

    Parameters
    ----------
    slices
        One :class:`volsurf.svi.SVIParams` per expiry; order does not matter.
    """

    def __init__(self, slices):
        ordered = sorted(slices, key=lambda p: p.expiry)
        if len(ordered) < 2:
            raise ValueError("need at least two expiries to differentiate in T")
        expiries = np.array([p.expiry for p in ordered], dtype=float)
        if np.any(np.diff(expiries) <= 0):
            raise ValueError("expiries must be distinct")
        self.slices = tuple(ordered)
        self.expiries = expiries

    # -- implied volatility side ------------------------------------------

    def _bracket(self, expiry):
        """Index of the slice interval containing ``expiry``, and the weight.

        Outside the quoted range the nearest interval is extrapolated, which
        keeps ``dw/dT`` constant rather than letting it drift to something the
        market never said.
        """
        expiry = np.asarray(expiry, dtype=float)
        idx = np.clip(
            np.searchsorted(self.expiries, expiry, side="right") - 1,
            0,
            len(self.expiries) - 2,
        )
        t0 = self.expiries[idx]
        t1 = self.expiries[idx + 1]
        return idx, (expiry - t0) / (t1 - t0), t1 - t0

    def _slice_values(self, k, idx, fn):
        """Evaluate ``fn`` on slices ``idx`` and ``idx + 1`` at the same ``k``."""
        k = np.asarray(k, dtype=float)
        lower = np.empty(np.broadcast(k, idx).shape, dtype=float)
        upper = np.empty_like(lower)
        idx_b = np.broadcast_to(idx, lower.shape)
        k_b = np.broadcast_to(k, lower.shape)
        for i in np.unique(idx_b):
            mask = idx_b == i
            lower[mask] = fn(self.slices[i], k_b[mask])
            upper[mask] = fn(self.slices[i + 1], k_b[mask])
        return lower, upper

    def total_variance(self, k, expiry):
        """Total implied variance, linear in ``T`` at fixed ``k``."""
        idx, weight, _ = self._bracket(expiry)
        lower, upper = self._slice_values(k, idx, total_variance)
        return lower + weight * (upper - lower)

    def implied_vol(self, k, expiry):
        """Black-Scholes implied volatility at log-moneyness ``k``."""
        expiry = np.asarray(expiry, dtype=float)
        return np.sqrt(np.maximum(self.total_variance(k, expiry), 0.0) / expiry)

    # -- local volatility side --------------------------------------------

    def _derivatives(self, k, expiry):
        idx, weight, gap = self._bracket(expiry)
        w_lo, w_hi = self._slice_values(k, idx, total_variance)
        k1_lo, k1_hi = self._slice_values(k, idx, d_total_variance_dk)
        k2_lo, k2_hi = self._slice_values(k, idx, d2_total_variance_dk2)
        w = w_lo + weight * (w_hi - w_lo)
        w_k = k1_lo + weight * (k1_hi - k1_lo)
        w_kk = k2_lo + weight * (k2_hi - k2_lo)
        w_t = (w_hi - w_lo) / gap
        return w, w_k, w_kk, w_t

    def local_variance(self, k, expiry):
        """Dupire local *variance* at log-moneyness ``k`` and time ``expiry``.

        Returns NaN where the surface is arbitrageable, rather than a clipped
        positive number: a negative local variance is a statement about the
        input prices, and silently flooring it would hide exactly the defect
        the caller needs to know about.  Use :meth:`diagnose` to find out which
        condition failed.
        """
        w, w_k, w_kk, w_t = self._derivatives(k, expiry)
        denom = dupire_denominator(w, w_k, w_kk, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            var = w_t / denom
        return np.where((denom > 0.0) & (w_t >= 0.0) & np.isfinite(var), var, np.nan)

    def local_vol(self, k, expiry):
        """Dupire local volatility; the square root of :meth:`local_variance`."""
        return np.sqrt(self.local_variance(k, expiry))

    def local_vol_from_spot(self, spot, forward, expiry):
        """Local volatility at an absolute level, given that expiry's forward.

        The simulation in :mod:`volsurf.mc` works in spot space, so it needs
        this rather than the log-moneyness form.
        """
        spot = np.asarray(spot, dtype=float)
        k = np.log(spot / np.asarray(forward, dtype=float))
        return self.local_vol(k, expiry)

    # -- diagnostics -------------------------------------------------------

    def diagnose(self, k_grid=None, tolerance=1e-10):
        """Check both no-arbitrage conditions over a grid of log-moneyness."""
        if k_grid is None:
            k_grid = np.linspace(-1.5, 1.5, 601)
        k_grid = np.asarray(k_grid, dtype=float)

        worst_g, worst_g_k = np.inf, float("nan")
        for params in self.slices:
            g = butterfly_g(params, k_grid)
            i = int(np.nanargmin(g))
            if g[i] < worst_g:
                worst_g, worst_g_k = float(g[i]), float(k_grid[i])

        worst_t, worst_t_k = np.inf, float("nan")
        for lower, upper in zip(self.slices, self.slices[1:]):
            gap = (total_variance(upper, k_grid) - total_variance(lower, k_grid)) / (
                upper.expiry - lower.expiry
            )
            i = int(np.nanargmin(gap))
            if gap[i] < worst_t:
                worst_t, worst_t_k = float(gap[i]), float(k_grid[i])

        return SurfaceDiagnosis(
            min_butterfly=worst_g,
            min_calendar=worst_t,
            worst_butterfly_k=worst_g_k,
            worst_calendar_k=worst_t_k,
            tolerance=tolerance,
        )
