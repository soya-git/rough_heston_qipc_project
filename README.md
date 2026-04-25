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
    alpha=0.6,
    lam=2.0,
    theta=0.04,
    rho=-0.5,
    nu=0.05,
    t=1.0,
)

model = RoughHestonModel(params)
price = model.calculate(NOuter=50, NInner=500, method="implicit")

print(f"price = {price:.12f}")
```

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

For a European call option, the implementation uses a Fourier representation of the payoff. After damping the payoff transform with the parameter $R$, the price is computed by numerical integration over Fourier nodes $u$:

$$
C(S_0, K, t) =
\frac{2 e^{-rt}}{2\pi}
\int_{0}^{\infty}
\mathrm{Re}\left(
L(u - iR) \, \widehat{g}(iR - u)
\right) \, du.
$$

Here $\mathrm{Re}(\cdot)$ means taking the real part of a complex number, $\widehat{g}$ is the transformed call payoff, and $L$ is the rough Heston characteristic-function term. The main numerical cost is evaluating $L$ at every Fourier node. This reduces to solving a fractional Riccati equation for an auxiliary function $h$:

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

#### `.calculate(NOuter, NInner, method="implicit", return_details=False)`

Method of a `RoughHestonModel` object. Call it on an initialized model instance to price one European call option:

```python
price = model.calculate(NOuter=50, NInner=500, method="implicit")
```

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

## License

MIT License. See `LICENSE` for details.
