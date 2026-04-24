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

The solver prices a European call option under a rough Heston-style characteristic-function representation. For each Fourier node, it solves a fractional Riccati equation of the form

```text
h(t) = I^alpha F(h)(t),     F(h) = A + B h + C h^2.
```

The original explicit Adams corrector has the endpoint update

```text
h_{k+1} = G_k + a F(h_{k+1}^P),
```

where `h_{k+1}^P` is the predictor and `G_k` is the historical Adams-Moulton contribution. The new method instead uses

```text
h_{k+1} = G_k + a F(h_{k+1}).
```

Because `F(h)` is quadratic, the implicit endpoint equation is a scalar complex quadratic equation:

```text
a C h^2 + (a B - 1) h + (G_k + a A) = 0.
```

Both roots are computed, and the root closest to the predictor is selected to maintain branch continuity.

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

### `RoughHestonModel.calculate(NOuter, NInner, method="quadratic_implicit", return_details=False)`

Prices one European call option.

Parameters:

- `NOuter: int` - number of Gauss-Legendre nodes for Fourier integration.
- `NInner: int` - number of time steps for the fractional Adams recursion. Must be even because Simpson quadrature is used for the final time integral.
- `method: str` - solver choice. Use `"quadratic_implicit"` for the new quadratic-implicit method or `"explicit"` for the original explicit predictor-corrector baseline.
- `return_details: bool` - if `True`, returns `(price, details)` instead of only `price`.

Returns:

- `float` if `return_details=False`.
- `(float, dict)` if `return_details=True`; details include Fourier nodes, weights, Riccati grid values, characteristic-function terms, and the selected method.

## Demo notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/rough-heston-qipc/blob/main/notebooks/demo.ipynb)

The notebook contains:

1. Package installation cell for Colab.
2. Basic price computation.
3. Runtime measurement.
4. Convergence visualisation as `NInner` increases.
5. Comparison with the explicit predictor-corrector baseline.

## Suggested project experiments

Use the demo notebook or `examples/benchmark.py` to evaluate:

1. **Convergence in time steps**: fix `NOuter` and vary `NInner`.
2. **Fourier quadrature convergence**: fix `NInner` and vary `NOuter`.
3. **Runtime comparison**: compare `method="quadratic_implicit"` and `method="explicit"` on the same model and grid.
4. **Accuracy-speed tradeoff**: use a high-resolution result as a reference and plot absolute error against runtime.
5. **Parameter stress tests**: vary `alpha`, `rho`, and `nu` to test robustness under roughness, leverage, and vol-of-vol changes.

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

## Publishing checklist

1. Replace `YOUR_GITHUB_USERNAME` in `README.md` and `pyproject.toml`.
2. Replace `your.email@example.com` in `pyproject.toml`.
3. Push the repository to GitHub.
4. Create a PyPI pending trusted publisher for project name `rough-heston-qipc`.
5. Create a GitHub environment named `pypi`.
6. Draft a GitHub release with a tag such as `v0.1.0`.
7. Publish the release to trigger `.github/workflows/publish.yml`.

## License

MIT License. See `LICENSE` for details.
