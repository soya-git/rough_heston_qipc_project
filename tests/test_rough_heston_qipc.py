import numpy as np
import pytest

from rough_heston_qipc import (
    RoughHestonParams,
    composite_simpson,
    gauss_legendre,
    quadratic_implicit_corrector,
    riccati_coefficients,
    rough_heston_explicit_pc,
    rough_heston_new,
    timed_price,
)


def test_gauss_legendre_integrates_constant():
    _, w = gauss_legendre(8, 0.0, 2.0)
    assert np.isclose(np.sum(w), 2.0)


def test_composite_simpson_requires_even_n():
    with pytest.raises(ValueError):
        composite_simpson(5, 0.0, 1.0)


def test_quadratic_implicit_corrector_residual_is_small():
    uL = np.array([0.25 - 1.5j, 1.0 - 1.5j], dtype=np.complex128)
    A, B, C = riccati_coefficients(uL, lam=2.0, rho=-0.5, nu=0.05)
    G = np.array([0.01 + 0.02j, -0.03 + 0.01j], dtype=np.complex128)
    predictor = G.copy()
    a_endpoint = 0.01

    h = quadratic_implicit_corrector(G, predictor, A, B, C, a_endpoint)
    Fh = A + B * h + C * h**2
    residual = h - (G + a_endpoint * Fh)

    assert np.max(np.abs(residual)) < 1e-10


def test_rough_heston_new_returns_finite_price():
    price = rough_heston_new(12, 20)
    assert np.isfinite(price)
    assert price >= 0.0


def test_return_details_shape():
    price, details = rough_heston_new(10, 12, return_details=True)
    assert np.isfinite(price)
    assert details["h"].shape == (10, 13)
    assert details["numF"].shape == (10, 13)
    assert details["u"].shape == (10,)


def test_new_and_explicit_are_close_on_small_grid():
    params = RoughHestonParams()
    p_new = rough_heston_new(12, 20, params=params)
    p_old = rough_heston_explicit_pc(12, 20, params=params)
    assert abs(p_new - p_old) < 5e-3


def test_timed_price_signature():
    price, elapsed = timed_price(10, 12)
    assert np.isfinite(price)
    assert elapsed >= 0.0
