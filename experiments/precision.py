"""How accurate is the Black formula, and compared to what?

Runs the measurement behind the central accuracy claim: for a few thousand
random points spanning the whole admissible domain, compare

* this library's :func:`volsurf.black.normalized_black`,
* the textbook formula evaluated directly in double precision,
* and the *conditioning* of the price at that point -- the factor by which the
  problem itself turns an ulp of input error into output error, and therefore
  the floor no double-precision routine can beat.

The point of plotting against conditioning rather than against moneyness is
that it separates the two things that get conflated: how hard the point is, and
how well the routine handles it.  A routine sitting on the diagonal is optimal;
one above it is losing digits of its own.

    python experiments/precision.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from volsurf.black import normalized_black

RESULTS = Path(__file__).resolve().parent / "results"
EPS = float(np.finfo(float).eps)


def textbook(x, s):
    """The formula as the textbooks write it, for comparison."""
    from scipy.special import ndtr

    h, t = x / s, s / 2
    return np.exp(0.5 * x) * ndtr(h + t) - np.exp(-0.5 * x) * ndtr(h - t)


def measure(n_points, seed=1234):
    import mpmath as mp

    mp.mp.dps = 50
    phi = lambda z: mp.erfc(-z / mp.sqrt(2)) / 2

    rng = np.random.default_rng(seed)
    xs = -np.exp(rng.uniform(np.log(1e-6), np.log(8.0), n_points))
    ss = np.exp(rng.uniform(np.log(1e-4), np.log(4.0), n_points))

    rows = []
    for x, s in zip(xs, ss):
        xm, sm = mp.mpf(float(x)), mp.mpf(float(s))
        h, t = xm / sm, sm / 2
        upper = mp.e ** (xm / 2) * phi(h + t)
        lower = mp.e ** (-xm / 2) * phi(h - t)
        exact = upper - lower
        if exact == 0 or exact < mp.mpf("1e-290"):
            continue
        vega = mp.e ** (-(h**2 + t**2) / 2) / mp.sqrt(2 * mp.pi)
        conditioning = float(
            sm * vega / exact + abs(xm) * (upper + lower) / (2 * exact)
        )

        ours = float(abs(mp.mpf(float(normalized_black(x, s))) / exact - 1)) / EPS
        naive_value = float(textbook(x, s))
        naive = (
            float(abs(mp.mpf(naive_value) / exact - 1)) / EPS
            if naive_value > 0
            else np.inf
        )
        rows.append(
            dict(
                x=float(x),
                s=float(s),
                price=float(exact),
                conditioning=max(conditioning, 1.0),
                ours_ulps=ours,
                textbook_ulps=naive,
            )
        )
    return rows


def plot(rows, path):
    from style import AQUA, BLUE, INK, MUTED, ORANGE, new_figure, save, style_axes

    cond = np.array([r["conditioning"] for r in rows])
    ours = np.array([r["ours_ulps"] for r in rows])
    naive = np.array([r["textbook_ulps"] for r in rows])
    finite = np.isfinite(naive)

    fig, axes = new_figure(1, 2, figsize=(11.5, 4.6))

    # -- error against conditioning ------------------------------------
    ax = style_axes(
        axes[0],
        "Error against the floor the problem imposes",
        "conditioning of the price at that point  (ulps)",
        "relative error  (ulps)",
    )
    floor = np.array([1.0, max(cond.max(), 10.0)])
    ax.plot(floor, floor, color=INK, lw=1.4, ls="--", zorder=4)
    ax.scatter(cond[finite], naive[finite], s=7, color=ORANGE, alpha=0.55,
               linewidths=0, zorder=2)
    ax.scatter(cond, np.maximum(ours, 1e-2), s=7, color=BLUE, alpha=0.6,
               linewidths=0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Direct labels rather than a legend box: identity is never colour-alone,
    # which also covers the aqua slot's contrast relief.
    ax.text(0.04, 0.93, "textbook formula", transform=ax.transAxes,
            color=ORANGE, fontsize=9.5, fontweight="bold")
    ax.text(0.04, 0.86, "this library", transform=ax.transAxes,
            color=BLUE, fontsize=9.5, fontweight="bold")
    ax.text(0.40, 0.15, "the floor: no double-precision routine\ncan get below this line",
            transform=ax.transAxes, color=INK, fontsize=8.5)

    # -- error against how small the price is ---------------------------
    ax = style_axes(
        axes[1],
        "Where the textbook formula gives out",
        "normalized option price",
        "relative error  (ulps)",
    )
    price = np.array([r["price"] for r in rows])
    ax.scatter(price[finite], naive[finite], s=7, color=ORANGE, alpha=0.55,
               linewidths=0)
    ax.scatter(price, np.maximum(ours, 1e-2), s=7, color=BLUE, alpha=0.6,
               linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # A normalized price cannot exceed one; without this the log axis pads out
    # to 1e6 and implies prices that do not exist.
    ax.set_xlim(right=2.0)
    ax.axhline(1.0, color=AQUA, lw=1.6, ls=":", zorder=4)
    ax.text(0.02, 0.16, "one ulp", transform=ax.transAxes, color=AQUA,
            fontsize=8.5, fontweight="bold")
    ax.text(0.04, 0.93, "textbook formula", transform=ax.transAxes,
            color=ORANGE, fontsize=9.5, fontweight="bold")
    ax.text(0.04, 0.86, "this library", transform=ax.transAxes,
            color=BLUE, fontsize=9.5, fontweight="bold")

    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="150 points instead of 1500, for a smoke test")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows = measure(150 if args.quick else 1500)

    cond = np.array([r["conditioning"] for r in rows])
    ours = np.array([r["ours_ulps"] for r in rows])
    naive = np.array([r["textbook_ulps"] for r in rows])
    finite = np.isfinite(naive)

    summary = dict(
        n_points=len(rows),
        ours_median_ulps=float(np.median(ours)),
        ours_p99_ulps=float(np.percentile(ours, 99)),
        ours_max_ulps=float(ours.max()),
        ours_worst_ratio_to_floor=float((ours / cond).max()),
        textbook_median_ulps=float(np.median(naive[finite])),
        textbook_p99_ulps=float(np.percentile(naive[finite], 99)),
        textbook_max_ulps=float(naive[finite].max()),
    )
    print(f"{'':34}{'median':>12}{'p99':>14}{'worst':>14}")
    print(f"{'this library (ulps)':34}{summary['ours_median_ulps']:>12.2f}"
          f"{summary['ours_p99_ulps']:>14.1f}{summary['ours_max_ulps']:>14.0f}")
    print(f"{'textbook formula (ulps)':34}{summary['textbook_median_ulps']:>12.2f}"
          f"{summary['textbook_p99_ulps']:>14.3g}{summary['textbook_max_ulps']:>14.3g}")
    print()
    print(f"worst case as a multiple of the conditioning floor: "
          f"{summary['ours_worst_ratio_to_floor']:.2f}x")

    (RESULTS / "precision.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.quick:
        plot(rows, RESULTS / "precision.png")


if __name__ == "__main__":
    main()
