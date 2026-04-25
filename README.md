# rough-heston-qipc

Rough Heston option pricing with a Quadratic-Implicit Fractional Adams Predictor-Corrector solver.

## What problem it solves

Rough Heston pricing requires solving a fractional Riccati equation inside a Fourier pricing integral. Standard fractional Adams predictor-corrector methods use an explicit corrector evaluation, which can be sensitive when the nonlinear Riccati term becomes stiff or when the time grid is coarse. This package implements a quadratic-implicit corrector that solves the nonlinear endpoint equation exactly at each Fourier node, while preserving the same overall fractional Adams recursion structure.

The project is designed for a computational finance course project: it provides a reusable Python package, a benchmark baseline, a Colab-ready demo notebook, tests, and GitHub Actions workflows for CI and PyPI publishing.

## Installation

After publishing to PyPI:

```bash
pip install rough-heston-qipc
```

For local development from the repository root:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from rough_heston_qipc import RoughHestonModel, RoughHestonParams, timed_price

model = RoughHestonModel(RoughHestonParams())
price = model.calculate(NOuter=50, NInner=500)
print(f"price = {price:.12f}")

price, elapsed = timed_price(NOuter=50, NInner=500)
print(f"price = {price:.12f}, elapsed = {elapsed:.4f} seconds")

params = RoughHestonParams(S0=1.0, K=1.0, alpha=0.6, rho=-0.5, nu=0.05)
custom_model = RoughHestonModel(params)
custom_price = custom_model.calculate(50, 500)
explicit_baseline = custom_model.calculate(50, 500, method="explicit")
print(custom_price)
```

## Method overview

The rough Heston model is a stochastic volatility model in which the variance process has rough, non-Markovian dynamics. Its volatility path is driven by a fractional kernel, so the current variance depends on the history of the process rather than only on the current state. This feature helps reproduce the steep short-maturity implied-volatility smiles observed in equity markets, but it also makes option pricing more numerically demanding than in the classical Heston model.

For a European call option, the implementation uses a Fourier representation of the payoff. After damping the payoff transform with the parameter $R$, the price is computed by numerical integration over Fourier nodes $u$:

$$
C(S_0, K, t) =
\frac{2 e^{-rt}}{2\pi}
\int_{0}^{\infty}
\operatorname{Re}\left(
L(u - iR) \, \widehat{g}(iR - u)
\right) \, du.
$$

Here $\widehat{g}$ is the transformed call payoff and $L$ is the rough Heston characteristic-function term. The main numerical cost is evaluating $L$ at every Fourier node. This reduces to solving a fractional Riccati equation for an auxiliary function $h$:

$$
h(t) = I^\alpha F(h)(t), \qquad F(h) = A + B h + C h^2.
$$

The fractional integral operator is

$$
I^\alpha f(t)
=
\frac{1}{\Gamma(\alpha)}
\int_0^t (t-s)^{\alpha - 1} f(s) \, ds,
$$

so the pricing problem ultimately depends on a nonlinear fractional differential or integral equation. The solver discretizes this equation on a time grid $t_k = k\Delta t$ and applies fractional Adams predictor-corrector recursions.

The first explicit component is the Adams-Bashforth predictor. It estimates the next value using only already-known history:

$$
h_{k+1}^{P}
=
\sum_{j=0}^{k} b_{k,j} F(h_j).
$$

The second explicit component is the standard Adams-Moulton corrector with the nonlinear endpoint evaluated at the predicted value:

$$
h_{k+1}
=
G_k + a F\left(h_{k+1}^{P}\right),
$$

where $G_k$ collects the historical Adams-Moulton terms and $a$ is the endpoint quadrature weight. This is the baseline exposed as `method="explicit"`.

The quadratic-implicit method keeps the same predictor and the same historical Adams-Moulton contribution, but evaluates the endpoint nonlinearity at the unknown corrected value:

$$
h_{k+1}
=
G_k + a F\left(h_{k+1}\right).
$$

Because $F(h)$ is quadratic, this implicit endpoint equation is still cheap to solve. At each time step and Fourier node it becomes a scalar complex quadratic equation:

$$
a C h^2 + (a B - 1)h + (G_k + a A) = 0.
$$

Both roots are computed, and the root closest to the Adams-Bashforth predictor is selected to maintain branch continuity. This is the default solver exposed as `method="implicit"`.

## API reference

### `RoughHestonParams`

Dataclass containing model and numerical parameters.

| Field | Type | Default | Description |
|---|---:|---:|---|
| `S0` | `float` | `1.0` | Initial stock price |
| `K` | `float` | `1.0` | Strike price |
| `r` | `float` | `0.0` | Risk-free rate |
| `z` | `float` | `0.4` | Initial variance/volatility input used in the source implementation |
| `alpha` | `float` | `0.6` | Fractional roughness parameter |
| `lam` | `float` | `2.0` | Mean-reversion speed |
| `theta` | `float` | `0.04` | Long-term variance level |
| `rho` | `float` | `-0.5` | Spot-volatility correlation |
| `nu` | `float` | `0.05` | Vol-of-vol parameter |
| `t` | `float` | `1.0` | Maturity |
| `R` | `float` | `1.5` | Fourier damping parameter |
| `u_lower` | `float` | `0.0` | Lower Fourier integration bound |
| `u_upper` | `float` | `25.0` | Upper Fourier integration bound |

### `RoughHestonModel(params=RoughHestonParams())`

Model object that stores one `RoughHestonParams` instance and exposes pricing through `calculate`.

### `RoughHestonModel.calculate(NOuter, NInner, method="implicit", return_details=False)`

Prices one European call option.

Parameters:

- `NOuter: int` - number of Gauss-Legendre nodes for Fourier integration.
- `NInner: int` - number of time steps for the fractional Adams recursion. Must be even because Simpson quadrature is used for the final time integral.
- `method: str` - solver choice. Use `"implicit"` for the new quadratic-implicit method or `"explicit"` for the original explicit predictor-corrector baseline.
- `return_details: bool` - if `True`, returns `(price, details)` instead of only `price`.

Returns:

- `float` if `return_details=False`.
- `(float, dict)` if `return_details=True`; details include Fourier nodes, weights, Riccati grid values, characteristic-function terms, and the selected method.

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soya-git/rough_heston_qipc_project/blob/main/notebooks/demo.ipynb)

The notebook contains:

1. Package installation cell for Colab.
2. Basic price computation.
3. Runtime measurement.
4. Convergence visualisation as `NInner` increases.
5. Comparison with the explicit predictor-corrector baseline.


## Repository structure

```text
rough-heston-qipc/
|-- rough_heston_qipc/
|   |-- __init__.py
|   `-- _core.py
|-- tests/
|   `-- test_rough_heston_qipc.py
|-- notebooks/
|   `-- demo.ipynb
|-- examples/
|   `-- benchmark.py
|-- .github/
|   `-- workflows/
|       |-- ci.yml
|       `-- publish.yml
|-- pyproject.toml
|-- README.md
`-- LICENSE
```

## License

MIT License. See `LICENSE` for details.
