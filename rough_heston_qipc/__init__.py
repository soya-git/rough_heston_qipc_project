"""Quadratic-Implicit Fractional Adams solver for Rough Heston option pricing.

The rough Heston characteristic-function model follows El Euch and Rosenbaum
(2019). The fractional Adams baseline follows Diethelm, Ford, and Freed
(2002, 2004).
"""

from ._core import (
    RoughHestonModel,
    RoughHestonParams,
    composite_simpson,
    fgc,
    gauss_legendre,
    quadratic_implicit_corrector,
    riccati_coefficients,
)

__all__ = [
    "RoughHestonModel",
    "RoughHestonParams",
    "composite_simpson",
    "fgc",
    "gauss_legendre",
    "quadratic_implicit_corrector",
    "riccati_coefficients",
]
