import numpy as np
import pytest

from rough_heston_qipc import (
    RoughHestonModel,
    RoughHestonParams,
    composite_simpson,
    gauss_legendre,
    quadratic_implicit_corrector,
    riccati_coefficients,
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


def test_model_calculate_returns_finite_price():
    model = RoughHestonModel(RoughHestonParams())
    price = model.calculate(12, 20)
    assert np.isfinite(price)
    assert price >= 0.0


def test_return_details_shape():
    model = RoughHestonModel()
    price, details = model.calculate(10, 12, return_details=True)
    assert np.isfinite(price)
    assert details["h"].shape == (10, 13)
    assert details["numF"].shape == (10, 13)
    assert details["u"].shape == (10,)
    assert details["method"] == "implicit"


def test_solver_methods_are_close_on_small_grid():
    model = RoughHestonModel(RoughHestonParams())
    p_new = model.calculate(12, 20, method="implicit")
    p_old = model.calculate(12, 20, method="explicit")
    assert abs(p_new - p_old) < 5e-3


def test_unknown_solver_method_raises():
    model = RoughHestonModel()
    with pytest.raises(ValueError):
        model.calculate(12, 20, method="unknown")
