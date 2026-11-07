"""The whole chain, end to end, and the one thing that limits it.

Dupire's construction says a volatility surface determines a unique diffusion
that reprices every European option on it.  That is a strong claim and it is
testable, but only if the test is genuinely closed:

    Heston  ->  prices  ->  implied vols  ->  SVI fit  ->  surface
            ->  Dupire local volatility   ->  Monte Carlo  ->  prices again

Every arrow is a different module, and the last one shares no algebra with the
others -- a Monte Carlo of the local volatility SDE cannot agree with the
surface by construction the way two Fourier rearrangements can.  Heston stands
in for a market because it gives an exact, arbitrage-free, *curved* smile at
every expiry, so the fit has something real to miss.

The run separates the two error sources instead of reporting one number:

* **fit error** -- SVI against the Heston smile, in volatility points.  SVI
  cannot represent Heston exactly, so this is never zero.
* **round-trip error** -- the Monte Carlo against the price the *fitted
  surface* implies.  This is what tests Dupire, and it is the number that
  should go to zero.

The headline finding is in the convergence study at the end.  With expiries
spaced as a market quotes them, the round trip misses by 1.5% -- and the cause
is not Dupire, the fit, or the simulation, but the piecewise-linear
interpolation of total variance between expiry slices.  ``dw/dT`` is the
numerator of the local variance, and interpolating linearly makes it a chord
slope where Heston's ``w(T)`` is curved.  Refining the expiry grid drives the
error to zero, and concentrating the refinement at the short end -- where the
curvature is -- is worth far more than adding slices uniformly.

    python experiments/round_trip.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from volsurf.black import forward_price
from volsurf.black import price as black_price
from volsurf.heston import HestonParams
from volsurf.heston import price as heston_price
from volsurf.implied import implied_vol
from volsurf.mc import price_european
from volsurf.surface import LocalVolSurface
from volsurf.svi import calibrate_slice, total_variance

RESULTS = Path(__file__).resolve().parent / "results"

# Equity-like: strong negative correlation, and the Feller condition violated,
# which is the usual state of calibrated parameters.
MODEL = HestonParams(v0=0.045, kappa=1.3, theta=0.05, xi=0.65, rho=-0.72)
SPOT, RATE, CARRY = 100.0, 0.025, 0.012
QUOTED_EXPIRIES = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)


def quote_smile(expiry, n_strikes=17):
    """Out-of-the-money Heston quotes at one expiry, as a market would show them.

    The strike range widens with ``sqrt(T)`` so each expiry covers a comparable
    number of standard deviations, which is how option chains are actually
    listed.
    """
    forward = float(forward_price(SPOT, RATE, CARRY, expiry))
    k = np.linspace(-0.45, 0.35, n_strikes) * max(np.sqrt(expiry), 0.45)
    strike = forward * np.exp(k)
    option_type = np.where(k > 0, 1, -1)
    price = np.array(
        [
            heston_price(MODEL, SPOT, float(s), expiry, RATE, CARRY, int(o))
            for s, o in zip(strike, option_type)
        ]
    )
    return k, strike, option_type, price


def build_surface(expiries, n_strikes=17):
    """Quote, invert, fit one SVI slice per expiry, assemble."""
    slices, fit_report = [], []
    for expiry in expiries:
        k, strike, option_type, price = quote_smile(expiry, n_strikes)
        quoted_vol = implied_vol(price, SPOT, strike, expiry, RATE, CARRY, option_type)
        params = calibrate_slice(k, quoted_vol**2 * expiry, expiry=expiry)
        fitted_vol = np.sqrt(total_variance(params, k) / expiry)
        slices.append(params)
        fit_report.append(
            dict(
                expiry=float(expiry),
                rmse_vol=float(np.sqrt(np.mean((fitted_vol - quoted_vol) ** 2))),
                max_vol_error=float(np.abs(fitted_vol - quoted_vol).max()),
            )
        )
    return LocalVolSurface(slices), fit_report


def reprice(surface, expiry, log_moneyness, n_paths, n_steps, seed=11):
    """Monte Carlo under the surface, against the price the surface implies."""
    forward = float(forward_price(SPOT, RATE, CARRY, expiry))
    strike = forward * np.exp(log_moneyness)
    option_type = 1 if log_moneyness >= 0 else -1

    result = price_european(
        surface, SPOT, strike, expiry, RATE, CARRY, option_type,
        n_paths=n_paths, n_steps=n_steps, seed=seed,
    )
    surface_vol = float(np.sqrt(surface.total_variance(log_moneyness, expiry) / expiry))
    surface_price = float(
        black_price(SPOT, strike, expiry, surface_vol, RATE, CARRY, option_type)
    )
    model_price = heston_price(
        MODEL, SPOT, float(strike), expiry, RATE, CARRY, option_type
    )
    return dict(
        expiry=float(expiry),
        log_moneyness=float(log_moneyness),
        monte_carlo=result.price,
        standard_error=result.standard_error,
        surface_price=surface_price,
        heston_price=float(model_price),
        round_trip_error=result.price - surface_price,
        sigmas=abs(result.price - surface_price) / max(result.standard_error, 1e-15),
        fit_error=surface_price - float(model_price),
        variance_reduction=result.variance_reduction,
    )


def expiry_convergence(counts, n_paths, n_steps):
    """How the round trip improves as the expiry grid is refined.

    Two spacings, because which one you choose matters more than how many
    slices you use.  ``sqrt``-spacing concentrates slices at the short end,
    where ``w(T)`` is most curved and a chord slope is worst.
    """
    expiry, log_moneyness = 0.5, 0.0
    rows = []
    for count in counts:
        for label, grid in (
            ("uniform", np.linspace(0.02, 2.0, count)),
            ("sqrt-spaced", np.linspace(np.sqrt(0.02), np.sqrt(2.0), count) ** 2),
        ):
            surface, _ = build_surface([float(t) for t in np.round(grid, 6)])
            row = reprice(surface, expiry, log_moneyness, n_paths, n_steps)
            row.update(n_slices=int(count), spacing=label)
            rows.append(row)
            print(
                f"  {label:<12} {count:>3} slices   "
                f"MC {row['monte_carlo']:.5f}   surface {row['surface_price']:.5f}   "
                f"error {100 * (row['monte_carlo'] / row['surface_price'] - 1):+.3f}%"
                f"   ({row['sigmas']:.1f} sigma)"
            )
    return rows


def plot(surface, fit_report, repricing, convergence, path):
    from style import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, new_figure, save, style_axes

    fig, axes = new_figure(1, 3, figsize=(15.5, 4.7))

    # -- smiles: quotes and fit ----------------------------------------
    ax = style_axes(
        axes[0],
        "SVI fitted to the Heston smile",
        "log-moneyness  log(K / F)",
        "implied volatility",
    )
    for colour, expiry in ((BLUE, 0.1), (ORANGE, 0.5), (AQUA, 2.0)):
        k, strike, option_type, price = quote_smile(expiry)
        quoted = implied_vol(price, SPOT, strike, expiry, RATE, CARRY, option_type)
        dense = np.linspace(k.min(), k.max(), 300)
        fitted = surface.implied_vol(dense, expiry)
        ax.plot(dense, fitted, color=colour, lw=2.0, zorder=3)
        ax.scatter(k, quoted, s=18, facecolor="none", edgecolor=colour,
                   linewidths=1.2, zorder=4)
        ax.text(dense[-1], fitted[-1], f"  {expiry:g}y", color=colour,
                fontsize=9, fontweight="bold", va="center")
    ax.text(0.03, 0.06, "circles: Heston quotes    lines: SVI fit",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)

    # -- the local volatility that comes out ---------------------------
    ax = style_axes(
        axes[1],
        "Local volatility extracted from it",
        "log-moneyness  log(K / F)",
        "volatility",
    )
    k = np.linspace(-0.4, 0.3, 250)
    for colour, expiry, style_ in ((BLUE, 0.5, "-"), (ORANGE, 1.5, "-")):
        ax.plot(k, surface.local_vol(k, expiry), color=colour, lw=2.0, ls=style_)
        ax.plot(k, surface.implied_vol(k, expiry), color=colour, lw=1.5, ls=":")
        ax.text(k[0], float(surface.local_vol(k[0], expiry)), f"{expiry:g}y  ",
                color=colour, fontsize=9, fontweight="bold", va="center", ha="right")
    ax.text(0.03, 0.10, "solid: local volatility", transform=ax.transAxes,
            color=INK, fontsize=8.5)
    ax.text(0.03, 0.04, "dotted: implied volatility", transform=ax.transAxes,
            color=MUTED, fontsize=8.5)

    # -- convergence in the expiry grid --------------------------------
    ax = style_axes(
        axes[2],
        "Round-trip error is set by the expiry grid",
        "number of expiry slices",
        "| Monte Carlo - surface price |  (% of price)",
    )
    for colour, label in ((ORANGE, "uniform"), (BLUE, "sqrt-spaced")):
        rows = [r for r in convergence if r["spacing"] == label]
        counts = [r["n_slices"] for r in rows]
        errors = [
            abs(100 * (r["monte_carlo"] / r["surface_price"] - 1)) for r in rows
        ]
        ax.plot(counts, errors, color=colour, lw=2.0, marker="o", ms=5.5)
        ax.text(counts[-1], errors[-1], f"  {label}", color=colour, fontsize=9,
                fontweight="bold", va="center")
    ax.set_yscale("log")
    ax.set_xscale("log")
    # Explicit ticks: "number of slices" in scientific notation reads badly.
    counts = sorted({r["n_slices"] for r in convergence})
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.minorticks_off()

    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small path counts and a short grid, for a smoke test")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    n_paths = 40_000 if args.quick else 400_000
    n_steps = 100 if args.quick else 400

    print("Building the surface from Heston quotes")
    surface, fit_report = build_surface(QUOTED_EXPIRIES)
    print(f"  {'expiry':>8} {'fit RMSE':>12} {'worst':>12}   (volatility points)")
    for row in fit_report:
        print(f"  {row['expiry']:>8.2f} {row['rmse_vol']:>12.2e} "
              f"{row['max_vol_error']:>12.2e}")
    diagnosis = surface.diagnose()
    print(f"  {diagnosis}")

    print("\nRepricing through the chain")
    grid = ((0.5, -0.15), (0.5, 0.0), (1.0, 0.0), (1.0, 0.15))
    if args.quick:
        grid = grid[:2]
    repricing = []
    print(f"  {'T':>5} {'k':>7} {'Monte Carlo':>14} {'surface':>12} "
          f"{'round trip':>12} {'fit':>10}")
    for expiry, k in grid:
        row = reprice(surface, expiry, k, n_paths, n_steps)
        repricing.append(row)
        print(f"  {row['expiry']:>5.2f} {row['log_moneyness']:>+7.2f} "
              f"{row['monte_carlo']:>14.5f} {row['surface_price']:>12.5f} "
              f"{row['round_trip_error']:>+12.5f} {row['fit_error']:>+10.5f}")

    print("\nConvergence in the expiry grid (0.5y at the money)")
    counts = (6, 13) if args.quick else (6, 11, 21, 41)
    convergence = expiry_convergence(counts, n_paths, n_steps)

    payload = dict(
        model=MODEL.__dict__,
        fit=fit_report,
        diagnosis=dict(
            min_butterfly=diagnosis.min_butterfly,
            min_calendar=diagnosis.min_calendar,
            arbitrage_free=diagnosis.arbitrage_free,
        ),
        repricing=repricing,
        convergence=convergence,
    )
    (RESULTS / "round_trip.json").write_text(json.dumps(payload, indent=2) + "\n")

    if not args.quick:
        plot(surface, fit_report, repricing, convergence, RESULTS / "round_trip.png")


if __name__ == "__main__":
    main()
