"""The Heston model: a characteristic function and three ways to price with it.

Heston gives stochastic volatility,

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho

and the reason it is the standard reference model is that although the density
of ``log S_T`` has no closed form, its *Fourier transform* does.  Everything
here follows from that one function.

Two things are easy to get wrong.

**The branch cut.**  The characteristic function involves ``sqrt`` and ``log``
of complex arguments, and the textbook (Heston 1993) arrangement crosses the
principal branch cut of the logarithm for perfectly ordinary parameters once
the maturity is long enough.  The symptom is vicious: prices are correct for
short maturities and then oscillate wildly, which looks like an integration
failure rather than a formula problem.  This module uses the Albrecher et al.
(2007) "Little Trap" arrangement, in which ``|g| < 1`` throughout and the
``log`` never wraps.  :func:`volsurf.heston.characteristic_function` documents
the specific sign choice, and the test suite covers 30-year maturities, where
the naive form visibly breaks.

**The scaling.**  The function returned is that of the *martingale-adjusted*
log return ``Y = log(S_T / F_T)``, so ``phi(0) = 1`` and ``phi(-i) = 1``
exactly.  That second identity is the discrete-time martingale condition and
is a sharp test of the whole expression; it is asserted directly.

Three pricing routes, deliberately kept separate so they can be checked
against each other:

:func:`price`
    Lewis' integral, one adaptive quadrature per option.  The reference.
:func:`price_fft`
    Carr-Madan, one FFT for a whole strip of strikes.  Hundreds of times
    faster per strike and agrees with the reference to ``1e-12`` absolute.
:func:`simulate`
    Monte Carlo with a full-truncation Euler scheme.  Slow and biased, and
    that is the point: it shares no code with the Fourier route, so agreement
    is real evidence rather than a consistency check between two rearrangements
    of the same algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .black import forward_price

__all__ = ["HestonParams", "characteristic_function", "price", "price_fft", "simulate"]


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters.

    ``v0`` is the initial instantaneous variance, ``kappa`` the mean-reversion
    speed, ``theta`` the long-run variance, ``xi`` the volatility of variance
    and ``rho`` the spot/vol correlation (negative for equities, which is what
    produces the skew).
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self):
        if self.v0 < 0 or self.theta < 0 or self.kappa <= 0 or self.xi <= 0:
            raise ValueError("v0, theta >= 0 and kappa, xi > 0 required")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")

    @property
    def feller(self):
        """``2 kappa theta / xi^2``; variance stays strictly positive if > 1.

        Market-calibrated equity parameters routinely violate this, which is
        why the Euler scheme in :func:`simulate` has to truncate rather than
        assume positivity.
        """
        return 2.0 * self.kappa * self.theta / self.xi**2


def characteristic_function(params: HestonParams, u, expiry):
    """``E[exp(i u log(S_T / F_T))]`` under the risk-neutral measure.

    The Little Trap arrangement: with

        d = sqrt((rho xi i u - kappa)^2 + xi^2 (i u + u^2))
        g = (kappa - rho xi i u - d) / (kappa - rho xi i u + d)

    the *minus* root in the numerator of ``g`` keeps ``|g| < 1`` for every
    ``u`` and every maturity, so ``log(1 - g exp(-d tau))`` stays on the
    principal branch.  Swapping to the plus root gives an algebraically
    identical expression that wraps the cut and produces oscillating garbage
    for maturities beyond a few years.
    """
    u = np.asarray(u)
    tau = np.asarray(expiry, dtype=float)
    iu = 1j * u

    kappa, theta, xi, rho, v0 = (
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
        params.v0,
    )

    rho_xi_iu = rho * xi * iu
    d = np.sqrt((rho_xi_iu - kappa) ** 2 + xi * xi * (iu + u * u))
    minus = kappa - rho_xi_iu - d
    g = minus / (kappa - rho_xi_iu + d)

    exp_dt = np.exp(-d * tau)
    denom = 1.0 - g * exp_dt

    big_c = (kappa * theta / (xi * xi)) * (
        minus * tau - 2.0 * np.log(denom / (1.0 - g))
    )
    big_d = (minus / (xi * xi)) * ((1.0 - exp_dt) / denom)
    return np.exp(big_c + big_d * v0)


def _lewis_integrand(params, u, log_moneyness, expiry):
    """Integrand of Lewis' representation, evaluated at ``u - i/2``."""
    phi = characteristic_function(params, u - 0.5j, expiry)
    return np.real(np.exp(1j * u * log_moneyness) * phi) / (u * u + 0.25)


def price(
    params: HestonParams,
    spot,
    strike,
    expiry,
    rate=0.0,
    carry=0.0,
    option_type=1,
    tol=1e-12,
):
    """Heston European price by Lewis' integral.

    Lewis' formula puts the whole strike dependence in a single oscillating
    factor and evaluates the characteristic function on the line
    ``Im(u) = -1/2``, where the integrand is real, even, and decays
    exponentially.  That symmetry is what makes one adaptive quadrature on
    ``[0, inf)`` enough:

        C / DF = F - sqrt(F K) / pi * int_0^inf
                 Re[exp(i u X) phi(u - i/2)] / (u^2 + 1/4) du,   X = log(F/K)

    Puts come from parity rather than a second integral, so the two are
    consistent by construction.  Scalar arguments only; use :func:`price_fft`
    for a strip.

    As ``xi -> 0`` with ``v0 = theta`` the variance freezes and the model
    becomes Black with volatility ``sqrt(v0)``.  Measured on a one-year
    at-the-money option, the approach has a clean order:

    * with ``rho = -0.7`` the gap is ``0.15 xi`` -- first order, because a
      correlated vol-of-vol produces skew at ``O(xi)``;
    * with ``rho = 0`` the gap is ``O(xi^2)``, since the leading term is the
      symmetric smile rather than the skew.

    Below ``xi`` of about ``1e-5`` floating-point cancellation takes over --
    ``D`` divides by ``xi^2`` a quantity that is itself ``O(xi^2)`` -- and the
    error stops falling.  Neither regime matters for calibrated parameters,
    where ``xi`` is of order one, and the frozen case has a closed form anyway.
    """
    spot, strike, expiry, rate, carry = (
        float(v) for v in (spot, strike, expiry, rate, carry)
    )
    fwd = float(forward_price(spot, rate, carry, expiry))
    discount = np.exp(-rate * expiry)
    log_moneyness = np.log(fwd / strike)

    integrand = lambda u: _lewis_integrand(params, u, log_moneyness, expiry)

    # The integrand decays exponentially, so integrating to a finite cutoff is
    # both faster and better conditioned than handing `quad` an infinite
    # interval (which otherwise reports roundoff at tight tolerances).  The
    # cutoff is found rather than assumed: double it until the integrand is
    # negligible against its value at the origin.
    scale = max(abs(integrand(0.0)), 1e-300)
    upper = 32.0
    while upper < 4096.0 and abs(integrand(upper)) > 1e-18 * scale:
        upper *= 2.0

    integral, _ = quad(integrand, 0.0, upper, epsabs=tol, epsrel=tol, limit=500)
    call = discount * (fwd - np.sqrt(fwd * strike) * integral / np.pi)
    if option_type in (1, "c", "call", "C"):
        return call
    return call - discount * (fwd - strike)


def price_fft(
    params: HestonParams,
    spot,
    expiry,
    rate=0.0,
    carry=0.0,
    n=4096,
    eta=0.25,
    alpha=1.5,
):
    """Carr-Madan FFT: one transform for a whole strip of strikes.

    Carr and Madan damp the call price by ``exp(alpha k)`` to make it square
    integrable, which turns the price into a single Fourier integral that the
    trapezoid rule and an FFT evaluate on a log-strike grid in one pass.  The
    grid is a consequence of the method rather than a choice -- the Nyquist
    relation ``d_k d_u = 2 pi / n`` ties strike spacing to frequency spacing --
    so this returns the strikes it computed rather than taking them as input.
    Interpolate, or use :func:`price` for specific strikes.

    ``alpha = 1.5`` is the usual damping choice; it must stay inside the strip
    where the characteristic function's second moment exists.

    Agrees with :func:`price` to ``1e-12`` *absolute* across maturities from
    one month to ten years, while pricing 4096 strikes in roughly the time the
    quadrature needs for one.  The bound is quoted in absolute terms on
    purpose: relative accuracy degrades for strikes whose option is worth less
    than ``1e-13``, which is a statement about the option, not the method.

    Returns
    -------
    strikes, calls
        Both length ``n``, ascending in strike.
    """
    fwd = float(forward_price(spot, rate, carry, expiry))
    discount = np.exp(-rate * float(expiry))

    u = np.arange(n) * eta
    d_k = 2.0 * np.pi / (n * eta)
    beta = np.log(fwd) - d_k * n / 2.0
    k_grid = beta + d_k * np.arange(n)

    # Characteristic function of log S_T, built from the log-return one.
    shifted = u - (alpha + 1.0) * 1j
    phi = characteristic_function(params, shifted, expiry) * np.exp(1j * shifted * np.log(fwd))
    psi = discount * phi / (alpha**2 + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u)

    # Plain trapezoid, and deliberately so.  Carr and Madan suggest Simpson
    # weights, and most implementations follow them, but measured against the
    # quadrature reference on this integrand Simpson is *worse* by seven orders
    # of magnitude: 4.8e-5 relative against 5.9e-12.
    #
    # The reason is the exponentially convergent trapezoidal rule.  On an
    # equispaced grid, Euler-Maclaurin expresses the trapezoid error entirely
    # in terms of odd derivatives at the endpoints; here the integrand is
    # analytic and decays to nothing at the cutoff, so every one of those terms
    # vanishes and convergence is geometric rather than algebraic.  Simpson's
    # rule gives up that structure for a fourth-order error term, which on this
    # integrand is strictly a downgrade.
    weights = np.full(n, eta)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    payload = psi * np.exp(-1j * beta * u) * weights
    calls = np.real(np.fft.fft(payload)) * np.exp(-alpha * k_grid) / np.pi
    return np.exp(k_grid), calls


def simulate(
    params: HestonParams,
    spot,
    expiry,
    n_paths=200_000,
    n_steps=400,
    rate=0.0,
    carry=0.0,
    seed=0,
    antithetic=True,
):
    """Terminal spot samples under a full-truncation Euler scheme.

    Market-calibrated equity parameters usually violate the Feller condition,
    so a discretised variance process *will* go negative and the scheme has to
    say what happens then.  Full truncation -- use ``max(v, 0)`` in both the
    drift and the diffusion but let the state itself stay negative -- has the
    smallest discretisation bias of the simple fixes (Lord et al., 2010) and,
    unlike reflection or absorption, does not quietly change the model's
    stationary distribution.

    The log-spot is stepped rather than the spot, which keeps positivity exact
    and removes the ``O(dt)`` bias an arithmetic scheme would add on top.
    Returns terminal spot values, which the caller averages into payoffs.
    """
    rng = np.random.default_rng(seed)
    dt = float(expiry) / n_steps
    sqrt_dt = np.sqrt(dt)

    base = n_paths // 2 if antithetic else n_paths
    drift = float(rate) - float(carry)

    log_s = np.full(base if not antithetic else 2 * base, np.log(float(spot)))
    var = np.full(log_s.shape, params.v0, dtype=float)
    rho = params.rho
    rho_bar = np.sqrt(max(1.0 - rho * rho, 0.0))

    for _ in range(n_steps):
        z1 = rng.standard_normal(base)
        z2 = rng.standard_normal(base)
        if antithetic:
            z1 = np.concatenate([z1, -z1])
            z2 = np.concatenate([z2, -z2])
        dw_v = z1 * sqrt_dt
        dw_s = (rho * z1 + rho_bar * z2) * sqrt_dt

        v_plus = np.maximum(var, 0.0)
        root_v = np.sqrt(v_plus)
        log_s += (drift - 0.5 * v_plus) * dt + root_v * dw_s
        var += params.kappa * (params.theta - v_plus) * dt + params.xi * root_v * dw_v

    return np.exp(log_s)
