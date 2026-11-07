"""volsurf -- derivatives pricing and volatility surfaces, built to the
precision limit.

The chain the library implements, in order:

``black`` / ``implied``
    The Black formula and its inverse, accurate to within a small factor of
    what double precision allows anywhere on the surface -- including the
    short-dated wings, where the textbook formula has no significant digits
    left at all.
``svi``
    Gatheral's SVI slice, calibrated by a reduction that makes three of the
    five parameters an exactly solvable bounded linear least squares, with the
    no-arbitrage conditions enforced rather than checked afterwards.
``surface``
    Slices assembled into a surface, and Dupire's local volatility extracted
    from it analytically.  The Dupire denominator turns out to be identically
    the butterfly-arbitrage function, so local volatility exists exactly when
    the surface is arbitrage-free.
``heston``
    A reference stochastic volatility model: characteristic function, Lewis
    quadrature, Carr-Madan FFT and Monte Carlo, each checked against the
    others.
``mc``
    Monte Carlo under a local volatility surface, which closes the loop by
    repricing the quotes the surface was built from.
"""

from .black import CALL, PUT, Greeks, greeks, normalized_black, price
from .heston import HestonParams
from .implied import implied_vol
from .mc import price_european
from .surface import LocalVolSurface
from .svi import SVIParams, calibrate_slice

__version__ = "0.1.0"

__all__ = [
    "CALL",
    "PUT",
    "Greeks",
    "HestonParams",
    "LocalVolSurface",
    "SVIParams",
    "calibrate_slice",
    "greeks",
    "implied_vol",
    "normalized_black",
    "price",
    "price_european",
    "__version__",
]
