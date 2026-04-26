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


def default_params(**overrides):
    values = {
        "S0": 100.0,
        "K": 100.0,
        "r": 0.0,
        "z": 0.4,
        "alpha": 0.6,
        "lam": 2.0,
        "theta": 0.04,
        "rho": -0.5,
        "nu": 0.05,
        "t": 1.0,
        "R": 1.5,
        "u_lower": 0.0,
        "u_upper": 25.0,
    }
    values.update(overrides)
    return RoughHestonParams(**values)


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


def test_model_price_returns_finite_price():
    model = RoughHestonModel(default_params())
    price = model.price(12, 20)
    assert np.isfinite(price)
    assert price >= 0.0


def test_return_details_shape():
    model = RoughHestonModel(default_params())
    price, details = model.price(10, 12, return_details=True)
    assert np.isfinite(price)
    assert details["h"].shape == (10, 13)
    assert details["numF"].shape == (10, 13)
    assert details["u"].shape == (10,)
    assert details["method"] == "implicit"
    assert details["option_type"] == "call"


def test_solver_methods_are_close_on_small_grid():
    model = RoughHestonModel(default_params())
    p_new = model.price(12, 20, method="implicit")
    p_old = model.price(12, 20, method="explicit")
    assert abs(p_new - p_old) < 5e-1


def test_put_price_satisfies_put_call_parity():
    params = default_params(r=0.03)
    model = RoughHestonModel(params)
    call_price = model.price(12, 20, option_type="call")
    put_price = model.price(12, 20, option_type="put")
    expected_put = call_price - params.S0 + params.K * np.exp(-params.r * params.t)
    assert np.isclose(put_price, expected_put)


def test_timed_price_returns_elapsed_seconds():
    model = RoughHestonModel(default_params())
    price, elapsed = model.price(10, 12, timed=True)
    assert np.isfinite(price)
    assert elapsed >= 0.0


def test_timed_return_details_includes_elapsed_seconds():
    model = RoughHestonModel(default_params())
    price, details = model.price(10, 12, timed=True, return_details=True)
    assert np.isfinite(price)
    assert details["elapsed"] >= 0.0


def test_price_returns_grid_when_params_contain_lists():
    model = RoughHestonModel(default_params(K=[90.0, 100.0], t=[0.5, 1.0]))
    rows = model.price(10, 12)
    assert len(rows) == 4
    assert {row["K"] for row in rows} == {90.0, 100.0}
    assert {row["t"] for row in rows} == {0.5, 1.0}
    assert all(np.isfinite(row["price"]) for row in rows)
    assert all(set(row) == {"K", "t", "price"} for row in rows)


def test_timed_grid_price_includes_elapsed_seconds():
    model = RoughHestonModel(default_params(K=[90.0, 100.0]))
    rows = model.price(10, 12, timed=True)
    assert len(rows) == 2
    assert all(row["elapsed"] >= 0.0 for row in rows)
    assert all(set(row) == {"K", "price", "elapsed"} for row in rows)


def test_grid_return_details_includes_details_per_row():
    model = RoughHestonModel(default_params(K=[100.0]))
    rows = model.price(10, 12, return_details=True)
    assert len(rows) == 1
    assert rows[0]["details"]["option_type"] == "call"
    assert set(rows[0]) == {"K", "price", "details"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("S0", 0.0),
        ("K", -1.0),
        ("z", -0.1),
        ("alpha", 0.5),
        ("alpha", 1.0),
        ("lam", 0.0),
        ("theta", -0.01),
        ("rho", -1.1),
        ("rho", 1.1),
        ("nu", -0.01),
        ("t", 0.0),
        ("R", 0.0),
    ],
)
def test_params_validate_invalid_scalar_values(field, value):
    with pytest.raises(ValueError):
        default_params(**{field: value})


def test_params_validate_invalid_list_values():
    with pytest.raises(ValueError):
        default_params(K=[90.0, -100.0])


def test_params_validate_fourier_bounds():
    with pytest.raises(ValueError):
        default_params(u_lower=25.0, u_upper=25.0)


def test_price_grid_returns_price_dataframe():
    model = RoughHestonModel(default_params())
    df = model.price_grid(NOuter_values=[8, 10], NInner_values=[10, 12])
    assert list(df.index) == [10, 12]
    assert list(df.columns) == [8, 10]
    assert df.index.name == "NInner"
    assert df.columns.name == "NOuter"
    assert np.isfinite(df.to_numpy()).all()


def test_timed_price_grid_returns_price_and_elapsed_dataframes():
    model = RoughHestonModel(default_params())
    price_df, elapsed_df = model.price_grid(NOuter_values=[8], NInner_values=[10, 12], timed=True)
    assert price_df.shape == (2, 1)
    assert elapsed_df.shape == (2, 1)
    assert np.isfinite(price_df.to_numpy()).all()
    assert (elapsed_df.to_numpy() >= 0.0).all()


def test_price_grid_rejects_list_valued_model_params():
    model = RoughHestonModel(default_params(K=[90.0, 100.0]))
    with pytest.raises(ValueError):
        model.price_grid(NOuter_values=[8], NInner_values=[10])


def test_unknown_solver_method_raises():
    model = RoughHestonModel(default_params())
    with pytest.raises(ValueError):
        model.price(12, 20, method="unknown")


def test_unknown_option_type_raises():
    model = RoughHestonModel(default_params())
    with pytest.raises(ValueError):
        model.price(12, 20, option_type="unknown")
