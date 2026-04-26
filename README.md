# Rough Heston QIPC

Rough Heston option pricing with a Quadratic-Implicit Fractional Adams Predictor-Corrector solver.

## What problem it solves

Rough Heston pricing requires solving a fractional Riccati equation inside a Fourier pricing integral. Standard fractional Adams predictor-corrector methods use an explicit corrector evaluation, which can be sensitive when the nonlinear Riccati term becomes stiff or when the time grid is coarse. This package implements a quadratic-implicit corrector that solves the nonlinear endpoint equation exactly at each Fourier node, while preserving the same overall fractional Adams recursion structure.

This package is designed to efficiently compute European option prices under the rough Heston model. It provides a reusable Python implementation with both explicit and implicit fractional Adams solvers, benchmark examples, a Colab-ready demo notebook, and tests for the core numerical routines.

## Installation

```bash
pip install rough-heston-qipc
```

## Quick start

```python
from rough_heston_qipc import RoughHestonModel, RoughHestonParams

params = RoughHestonParams(
    S0=100.0,
    K=100.0,
    r=0.0,
    z=0.4,
    alpha=0.6,
    lam=2.0,
    theta=0.04,
    rho=-0.5,
    nu=0.05,
    t=1.0,
    R=1.5,
    u_lower=0.0,
    u_upper=25.0,
)

model = RoughHestonModel(params)
price = model.price(
    NOuter=50,
    NInner=500,
    method="implicit",
    option_type="call",
)

print(f"price = {price:.12f}")
```

To price a grid, pass lists directly in `RoughHestonParams`:

```python
grid_params = RoughHestonParams(
    S0=100.0,
    K=[90.0, 100.0, 110.0],
    r=0.0,
    z=0.4,
    alpha=0.6,
    lam=2.0,
    theta=0.04,
    rho=-0.5,
    nu=0.05,
    t=[0.5, 1.0],
    R=1.5,
    u_lower=0.0,
    u_upper=25.0,
)
grid_model = RoughHestonModel(grid_params)
rows = grid_model.price(NOuter=50, NInner=500, timed=True)
```

Each grid row contains only the list-valued parameters, `price`, and `elapsed` when `timed=True`.

## Method overview

The rough Heston model is a stochastic volatility model where the stock price follows

$$
dS_t = S_t \sqrt{V_t}\, dW_t
$$

and the variance is driven by a fractional kernel:

$$
V_t = V_0 + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha - 1} \lambda(\theta - V_s)\, ds + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha - 1} \nu \sqrt{V_s}\, dB_s.
$$

The fractional kernel, with $\alpha \in (1/2, 1)$ and Brownian correlation $\rho$, makes the variance path rough and history-dependent, which helps capture short-maturity implied-volatility smiles but turns option pricing into a fractional Riccati problem.

For a European option, the implementation uses a Fourier representation of the call payoff. After damping the payoff transform with the parameter $R$, the call price is computed by numerical integration over Fourier nodes $u$:

$$
C(S_0, K, t) =
\frac{2 e^{-rt}}{2\pi}
\int_{0}^{\infty}
\mathrm{Re}\left(
L(u - iR) \, \widehat{g}(iR - u)
\right) \, du.
$$

Here $\mathrm{Re}(\cdot)$ means taking the real part of a complex number, $\widehat{g}$ is the transformed call payoff, and $L$ is the rough Heston characteristic-function term. Put prices are obtained from the computed call price by put-call parity. The main numerical cost is evaluating $L$ at every Fourier node. This reduces to solving a fractional Riccati equation for an auxiliary function $h$:

$$
h(t) = I^\alpha F(h)(t), \qquad F(h) = A + B h + C h^2.
$$

The fractional integral operator is

$$
I^\alpha f(t) =
\frac{1}{\Gamma(\alpha)}
\int_0^t (t-s)^{\alpha - 1} f(s) \, ds,
$$

so the pricing problem ultimately depends on a nonlinear fractional differential or integral equation. The solver discretizes this equation on a time grid $t_k = k\Delta t$ and applies fractional Adams predictor-corrector recursions.

The first explicit component is the Adams-Bashforth predictor. It estimates the next value using only already-known history:

$$
h_{k+1}^{P} =
\sum_{j=0}^{k} b_{k,j} F(h_j).
$$

The second explicit component is the standard Adams-Moulton corrector with the nonlinear endpoint evaluated at the predicted value:

$$
h_{k+1} =
G_k + a F\left(h_{k+1}^{P}\right),
$$

where $G_k$ collects the historical Adams-Moulton terms and $a$ is the endpoint quadrature weight. This is the baseline exposed as `method="explicit"`.

The quadratic-implicit method keeps the same predictor and the same historical Adams-Moulton contribution, but evaluates the endpoint nonlinearity at the unknown corrected value:

$$
h_{k+1} =
G_k + a F\left(h_{k+1}\right).
$$

Because $F(h)$ is quadratic, this implicit endpoint equation is still cheap to solve. At each time step and Fourier node it becomes a scalar complex quadratic equation:

$$
a C h^2 + (a B - 1)h + (G_k + a A) = 0.
$$

Both roots are computed, and the root closest to the Adams-Bashforth predictor is selected to maintain branch continuity. This is the default solver exposed as `method="implicit"`.

## API reference

### `RoughHestonParams`

Dataclass containing model and numerical parameters. Each field accepts either a scalar value or a list of values. If any field is a list, `RoughHestonModel.price` automatically prices every combination of list-valued parameters and returns grid rows. Parameter validity is checked during initialization.

| Field | Type | Default | Description |
|---|---:|---:|---|
| `S0` | `float` or `list[float]` | Required | Initial stock price |
| `K` | `float` or `list[float]` | Required | Strike price |
| `r` | `float` or `list[float]` | Required | Risk-free rate |
| `z` | `float` or `list[float]` | Required | Initial variance/volatility input used in the source implementation |
| `alpha` | `float` or `list[float]` | Required | Fractional roughness parameter |
| `lam` | `float` or `list[float]` | Required | Mean-reversion speed |
| `theta` | `float` or `list[float]` | Required | Long-term variance level |
| `rho` | `float` or `list[float]` | Required | Spot-volatility correlation |
| `nu` | `float` or `list[float]` | Required | Vol-of-vol parameter |
| `t` | `float` or `list[float]` | Required | Maturity |
| `R` | `float` or `list[float]` | Required | Fourier damping parameter |
| `u_lower` | `float` or `list[float]` | Required | Lower Fourier integration bound |
| `u_upper` | `float` or `list[float]` | Required | Upper Fourier integration bound |

Validation requires `S0 > 0`, `K > 0`, `z >= 0`, `alpha in (0.5, 1.0)`, `lam > 0`, `theta >= 0`, `rho in [-1.0, 1.0]`, `nu >= 0`, `t > 0`, `R > 0`, and `u_upper > u_lower`.

### `RoughHestonModel(params)`

Model object that stores one `RoughHestonParams` instance and exposes pricing through `price` and numerical grid experiments through `price_grid`.

#### `.price(NOuter, NInner, method="implicit", option_type="call", timed=False, return_details=False)`

Method of a `RoughHestonModel` object. Call it on an initialized model instance to price one European option:

```python
price = model.price(
    NOuter=50,
    NInner=500,
    method="implicit",
    option_type="call",
)
```

Parameters:

- `NOuter: int` - number of Gauss-Legendre nodes for Fourier integration.
- `NInner: int` - number of time steps for the fractional Adams recursion. Must be even because Simpson quadrature is used for the final time integral.
- `method: str` - solver choice. Use `"implicit"` for the new quadratic-implicit method or `"explicit"` for the original explicit predictor-corrector baseline.
- `option_type: str` - payoff type. Use `"call"` or `"put"`.
- `timed: bool` - if `True`, also returns the elapsed time for the computation.
- `return_details: bool` - if `True`, returns `(price, details)` instead of only `price`.

Returns:

- `float` if `timed=False` and `return_details=False`.
- `(float, float)` as `(price, elapsed)` if `timed=True` and `return_details=False`.
- `(float, dict)` if `return_details=True`; details include Fourier nodes, weights, Riccati grid values, characteristic-function terms, and the selected method.
- `list[dict]` if any `RoughHestonParams` field is a list. Each row contains only the list-valued parameter names and their current scalar values, plus `price`; if `timed=True`, each row also contains `elapsed`.

#### `.price_grid(NOuter_values, NInner_values, method="implicit", option_type="call", timed=False)`

Prices every combination of `NOuter_values` and `NInner_values` and returns a pandas DataFrame indexed by `NInner`, with `NOuter` values as columns:

```python
price_df = model.price_grid(
    NOuter_values=[25, 50, 100],
    NInner_values=[100, 200, 500],
)

price_df, elapsed_df = model.price_grid(
    NOuter_values=[25, 50, 100],
    NInner_values=[100, 200, 500],
    timed=True,
)
```

If `timed=False`, the method returns one DataFrame of prices. If `timed=True`, it returns `(price_df, elapsed_df)`. `price_grid` requires all `RoughHestonParams` fields to be scalar values; if any parameter is a list, it raises `ValueError`.

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soya-git/rough_heston_qipc_project/blob/main/notebooks/demo.ipynb)

The notebook contains:

1. Package installation cell for Colab.
2. Basic price computation.
3. Runtime measurement.
4. Convergence visualisation as `NInner` increases.
5. Comparison with the explicit predictor-corrector baseline.

## License

MIT License. See `LICENSE` for details.
