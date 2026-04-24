"""Quadratic-Implicit Fractional Adams solver for Rough Heston option pricing."""

from ._core import (
    RoughHestonParams,
    composite_simpson,
    fgc,
    gauss_legendre,
    quadratic_implicit_corrector,
    riccati_coefficients,
    rough_heston_explicit_pc,
    rough_heston_new,
    run_grid_test_new,
    timed_price,
)

__all__ = [
    "RoughHestonParams",
    "composite_simpson",
    "fgc",
    "gauss_legendre",
    "quadratic_implicit_corrector",
    "riccati_coefficients",
    "rough_heston_explicit_pc",
    "rough_heston_new",
    "run_grid_test_new",
    "timed_price",
]
