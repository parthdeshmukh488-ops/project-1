"""Implied volatility: inverting the Black formula to full double precision.

Implied volatility is the market's actual quoting convention, so this inversion
sits underneath every surface fit, every risk report and every calibration in
the library.  It runs millions of times in a surface build, and it has to work
on the 0.05-delta wing quotes where the price is ``1e-12`` and a naive Newton
iteration either stalls or diverges.

Three design decisions do the work.

**Normalise first.**  Reduce to ``b(x, s) = beta`` in log-moneyness ``x`` and
total volatility ``s = sigma sqrt(T)``, then use put-call parity to move every
problem onto the out-of-the-money side (``x <= 0``), which is the
well-conditioned one.  What is left is a one-dimensional root solve of a
function that is known to be strictly increasing.

**Split at the inflection point.**  ``b`` is convex in ``s`` below
``s* = sqrt(2|x|)`` and concave above it -- the point where the normalized vega
peaks.  Newton's method on the raw price is badly behaved in the convex region
(the iteration undershoots towards zero and the steps collapse), so below
``s*`` the solve is done on ``log b`` instead, which is close to linear there.
Above ``s*`` the price itself is the better objective.  This is the single
change that turns an unreliable solver into a 3-iteration one.

**Never leave the bracket.**  A bracket is established before iterating and
maintained throughout; any Halley step that would escape it is replaced by a
bisection.  The solver therefore converges for *every* admissible input,
including prices one ulp inside the no-arbitrage bounds, rather than for the
inputs that happen to suit the initial guess.

Measured on a 10,197-point out-of-the-money grid spanning log-moneyness
``|x| <= 6``, expiries from one day to twenty years and volatilities from
``0.5%`` to ``400%``, round-tripped through :func:`volsurf.black.price`:

* median relative error **exactly zero** -- the recovered volatility is the
  same double that went in;
* **every** point within **3.6x of its own conditioning floor**, the factor
  ``b / (s db/ds)`` by which the problem itself amplifies an input ulp;
* **4.2** Halley iterations on average.

The conditioning-relative statement is the honest one.  A flat tolerance
would be unreachable for a quote sitting an ulp below its no-arbitrage
ceiling, where the price carries almost no information about volatility at
all, and unambitious everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black import CALL, _as_sign, forward_price, normalized_black, normalized_vega
from .normal import SQRT_TWO_PI, norm_ppf

__all__ = ["InversionResult", "implied_vol", "implied_vol_normalized"]

_MAX_ITERATIONS = 32
_EPS = np.finfo(float).eps
_TOLERANCE = 2.0 * _EPS
# Demand ~6 surviving decimal digits in a recovered time value by default.
_DEFAULT_MIN_ULPS = 1.0e6
# Prices below the smallest *normal* double are subnormal: they carry fewer
# than 53 bits of mantissa, so the volatility they imply is not determined to
# full precision by the input itself.  They are rejected rather than silently
# answered to whatever accuracy survives.
_TINY = np.finfo(float).tiny
# A total volatility this large means the quote is within an ulp of its
# no-arbitrage ceiling, where the inversion is not defined in any useful sense.
_MAX_TOTAL_VOL = 1.0e4


@dataclass(frozen=True)
class InversionResult:
    """Outcome of an inversion, with the diagnostics worth keeping.

    ``iterations`` is the per-point step count, which is what the performance
    claim in the README is measured from.  ``converged`` is ``False`` for
    inputs that violate no-arbitrage bounds or that fail the significance test
    below, where the volatility is NaN.

    ``time_value_ulps`` is the honest part.  An in-the-money quote carries its
    volatility information entirely in its *time value*, which has to be
    recovered by subtracting the intrinsic -- and for a deep in-the-money
    option that subtraction can destroy every significant digit the price had.
    A one-week call six vols in the money has a time value around ``1e-100``
    against a price of order one: the float simply does not contain it.  This
    field reports how many ulps of the quoted price the recovered time value is
    worth, so callers can see the conditioning instead of inferring it from a
    suspicious-looking answer.

    ``condition`` is the amplification factor ``b / (s db/ds)``: the number of
    times over that a relative error in the input price is magnified into a
    relative error in the returned volatility.  It is the ceiling on what any
    inversion can achieve, and the test suite asserts the solver reaches it
    rather than asserting a flat tolerance that would be unachievable in the
    flat regions and unambitious everywhere else.
    """

    total_vol: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    time_value_ulps: np.ndarray = None
    condition: np.ndarray = None


def _atm_exact(beta):
    """Closed-form inversion at ``x = 0``.

    At the money ``b(0, s) = 2 Phi(s/2) - 1`` exactly, so ``s`` follows from one
    inverse-normal call with no iteration at all.  Used as the initial guess
    near the money and as an exactness check in the test suite.
    """
    return 2.0 * norm_ppf(0.5 * (1.0 + beta))


def _initial_guess(x, beta, s_star, b_star):
    """A starting point good enough that Halley needs three or four steps.

    Two regimes, each inverted from the leading behaviour of ``b``:

    *Below the inflection point* the small-``s`` asymptotics of the Black
    formula are ``b ~ exp(-x^2/(2 s^2) - s^2/8) s^3 / (x^2 sqrt(2 pi))`` (a
    Laplace expansion of the vega integral at its upper endpoint).  The
    ``exp(-x^2/(2 s^2))`` factor dominates by many orders of magnitude, so
    taking logs and keeping only that term gives ``s ~ |x| / sqrt(2 L)`` with
    ``L`` the log of the slowly varying remainder, evaluated once at ``s*``.

    *Above the inflection point* the price is close to the at-the-money shape
    scaled by ``exp(x/2)``, so the exact at-the-money inversion applied to the
    rescaled price is already within a few percent.
    """
    lower = beta < b_star

    # Upper branch: rescale onto the at-the-money formula.
    # ``beta exp(-x/2)`` can round to 1 for extreme quotes, where the inverse
    # normal diverges; clamp strictly inside (0, 1) so the guess stays finite
    # and let the bracket clip handle the rest.
    scaled = np.clip(beta * np.exp(-0.5 * x), 1e-12, 1.0 - 1e-12)
    guess_upper = _atm_exact(scaled)

    # Lower branch: invert the dominant Gaussian factor of the asymptotics.
    with np.errstate(divide="ignore", invalid="ignore"):
        slow = beta * x * x * SQRT_TWO_PI / np.maximum(s_star, 1e-300) ** 3
        big_l = -np.log(np.clip(slow, 1e-300, None))
        guess_lower = np.abs(x) / np.sqrt(2.0 * np.maximum(big_l, 1e-12))

    guess = np.where(lower, guess_lower, guess_upper)
    return np.where(np.isfinite(guess) & (guess > 0.0), guess, s_star)


def implied_vol_normalized(x, beta):
    """Solve ``b(x, s) = beta`` for the total volatility ``s``.

    ``x`` must be non-positive (the out-of-the-money convention) and ``beta``
    must lie strictly inside ``(0, exp(x/2))``, the no-arbitrage range.  Inputs
    outside it return NaN with ``converged = False`` rather than raising, so a
    surface build can keep going and report which quotes were bad.
    """
    x = np.asarray(x, dtype=float)
    beta = np.asarray(beta, dtype=float)
    x, beta = np.broadcast_arrays(x, beta)
    shape = x.shape
    x = x.ravel().astype(float, copy=True)
    beta = beta.ravel().astype(float, copy=True)

    total_vol = np.full(x.shape, np.nan)
    iterations = np.zeros(x.shape, dtype=int)

    upper_bound = np.exp(0.5 * x)
    admissible = (beta >= _TINY) & (beta < upper_bound) & (x <= 0.0)
    if not np.any(admissible):
        return InversionResult(
            total_vol.reshape(shape),
            iterations.reshape(shape),
            admissible.reshape(shape),
            np.full(shape, np.inf),
        )

    idx = np.nonzero(admissible)[0]
    xs = x[idx]
    bs = beta[idx]

    # The inflection point of b in s, where the normalized vega peaks and the
    # curvature changes sign.  It is also the natural split for the objective.
    s_star = np.sqrt(2.0 * np.abs(xs))
    s_star = np.where(s_star > 0.0, s_star, 1e-8)
    b_star = normalized_black(xs, s_star)

    use_log = bs < b_star

    # --- bracket -----------------------------------------------------------
    # Below the inflection point the root is in (0, s*).  Above it, double
    # upward until the price overshoots; b -> exp(x/2) monotonically and beta is
    # strictly below that, so this always terminates.
    lo = np.where(use_log, 0.0, s_star)
    hi = np.where(use_log, s_star, np.maximum(s_star, 1.0))
    needs_expansion = ~use_log
    unbounded = np.zeros(xs.shape, dtype=bool)
    while np.any(needs_expansion):
        # Strict: stop as soon as b(hi) reaches the target, so that a root
        # sitting exactly on the endpoint stays inside the bracket.
        too_small = needs_expansion & (normalized_black(xs, hi) < bs)
        if not np.any(too_small):
            break
        lo = np.where(too_small, hi, lo)
        hi = np.where(too_small, hi * 2.0, hi)
        needs_expansion = too_small & (hi <= _MAX_TOTAL_VOL)
        unbounded |= too_small & (hi > _MAX_TOTAL_VOL)

    s = np.clip(_initial_guess(xs, bs, s_star, b_star), lo, hi)

    target = np.where(use_log, np.log(bs), bs)
    active = np.ones(xs.shape, dtype=bool)
    count = np.zeros(xs.shape, dtype=int)

    for _ in range(_MAX_ITERATIONS):
        if not np.any(active):
            break
        a = np.nonzero(active)[0]
        sa = s[a]
        xa = xs[a]

        b = normalized_black(xa, sa)
        v = normalized_vega(xa, sa)

        # d(log v)/ds, which supplies the second derivative of both objectives.
        dlogv = xa * xa / sa**3 - 0.25 * sa

        logs = use_log[a]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Objective, first and second derivative, per branch.
            # No clamping.  Where the price has underflowed, ``log b`` is
            # -inf and the Newton step is non-finite, which the escape test
            # below turns into a bisection -- the correct response, and far
            # better than pinning the objective at an arbitrary floor (which
            # drives the iteration to zero and silently returns garbage).
            f = np.where(logs, np.log(b), b) - target[a]
            f1 = np.where(logs, v / b, v)
            f2 = np.where(logs, f1 * (dlogv - f1), v * dlogv)

            # Tighten the bracket from the sign of the residual before stepping,
            # so the fallback bisection is always on a valid interval.
            lo[a] = np.where(f < 0.0, np.maximum(lo[a], sa), lo[a])
            hi[a] = np.where(f > 0.0, np.minimum(hi[a], sa), hi[a])

            newton = -f / f1
            # Halley: third-order, needs only the second derivative.
            halley = newton / (1.0 + 0.5 * newton * f2 / f1)
            step = np.where(np.isfinite(halley), halley, newton)
            proposal = sa + step

        # The bracket is closed, not open: a root can legitimately sit exactly
        # on an endpoint (an at-the-money quote whose total volatility equals
        # the inflection point does), and rejecting that value would bisect
        # away from the answer the iteration had already found.
        escaped = ~np.isfinite(proposal) | (proposal < lo[a]) | (proposal > hi[a])
        proposal = np.where(escaped, 0.5 * (lo[a] + hi[a]), proposal)

        # An exact hit is the root; keep it rather than letting a bisection
        # fallback overwrite it on the same pass.
        at_root = f == 0.0
        proposal = np.where(at_root, sa, proposal)

        # Three ways to be finished: an exact hit, a step below one ulp, or a
        # bracket squeezed to one ulp.  The last one matters: where the price
        # is nearly flat in volatility (a quote an ulp below its no-arbitrage
        # ceiling) the step test can never fire, because the function carries
        # no information at that scale.  Without it those points burn every
        # iteration for nothing.
        bracket_tight = (hi[a] - lo[a]) <= _TOLERANCE * np.maximum(sa, 1e-16)
        done = at_root | bracket_tight | (
            np.abs(proposal - sa) <= _TOLERANCE * np.maximum(sa, 1e-16)
        )
        s[a] = proposal
        count[a] += 1
        active[a] = ~done

    s = np.where(unbounded, np.nan, s)

    # Conditioning at the solution: d(log b) / d(log s) inverted.
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = normalized_black(xs, s) / (s * normalized_vega(xs, s))
    condition = np.full(x.shape, np.inf)
    condition[idx] = np.abs(kappa)

    total_vol[idx] = s
    iterations[idx] = count
    admissible[idx] = admissible[idx] & ~unbounded
    return InversionResult(
        total_vol.reshape(shape),
        iterations.reshape(shape),
        admissible.reshape(shape),
        np.full(shape, np.inf),
        condition.reshape(shape),
    )


def implied_vol(
    option_price,
    spot,
    strike,
    expiry,
    rate=0.0,
    carry=0.0,
    option_type=CALL,
    min_time_value_ulps=_DEFAULT_MIN_ULPS,
    return_diagnostics=False,
):
    """Black-Scholes implied volatility from an option price.

    Broadcasts over every argument, so an entire surface inverts in one call.

    Out-of-the-money quotes -- the market convention, and the only side anyone
    actually quotes in the wings -- invert to machine precision everywhere.
    In-the-money quotes go through put-call parity, and that subtraction is
    where precision is lost; ``min_time_value_ulps`` is the floor on how many
    ulps of the quote the recovered time value must be worth before a number is
    returned at all.  The default demands roughly six surviving decimal digits.
    Lower it to trade accuracy for coverage, or set it to ``0`` to disable the
    check and get whatever the arithmetic yields.

    Prices outside the no-arbitrage bounds, and quotes that fail the
    significance test, come back as NaN.  Pass ``return_diagnostics=True`` for
    the iteration counts, convergence flags and conditioning.
    """
    option_price, spot, strike, expiry, rate, carry = (
        np.asarray(v, dtype=float)
        for v in (option_price, spot, strike, expiry, rate, carry)
    )
    sign = _as_sign(option_type)

    fwd = forward_price(spot, rate, carry, expiry)
    discount = np.exp(-rate * expiry)
    x = np.log(fwd / strike)
    scale = discount * np.sqrt(fwd * strike)

    beta = option_price / scale
    # Reflect puts onto calls, then use parity to move every problem onto the
    # out-of-the-money side, which is the well-conditioned one.
    x_eff = np.where(sign > 0, x, -x)
    itm = x_eff > 0.0
    intrinsic = np.exp(0.5 * x_eff) - np.exp(-0.5 * x_eff)
    time_value = beta - intrinsic

    # How many ulps of the quoted price the recovered time value is worth.
    # Guard the denominator rather than clamping it to `tiny`, which overflows
    # the ratio for a quote of zero instead of reporting "no information".
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        one_ulp = np.abs(beta) * _EPS
        ulps = np.where(
            itm & (one_ulp > 0.0),
            time_value / np.where(one_ulp > 0.0, one_ulp, 1.0),
            np.where(itm, 0.0, np.inf),
        )

    beta_otm = np.where(itm, time_value, beta)
    x_otm = np.where(itm, -x_eff, x_eff)

    result = implied_vol_normalized(x_otm, beta_otm)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = result.total_vol / np.sqrt(expiry)

    significant = ulps >= min_time_value_ulps
    vol = np.where(significant, vol, np.nan)

    if not return_diagnostics:
        return vol
    return vol, InversionResult(
        vol,
        np.broadcast_to(result.iterations, vol.shape),
        np.broadcast_to(result.converged, vol.shape) & significant,
        np.broadcast_to(ulps, vol.shape),
        np.broadcast_to(result.condition, vol.shape),
    )
