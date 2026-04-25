"""Rough Heston option pricing.

Rough Heston option pricing with a Quadratic-Implicit Fractional Adams
Predictor-Corrector solver.

Main entry points
-----------------
RoughHestonModel(params).calculate(NOuter, NInner, method="implicit")
    Price one European call option under a Rough Heston-style characteristic
    function representation.

run_grid_test_new()
    Python equivalent of test.m, using the default model solver.

This file is self-contained except for NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma
from typing import Literal, Tuple
import time

import numpy as np


SolverMethod = Literal["implicit", "explicit"]


@dataclass(frozen=True)
class RoughHestonParams:
    """Model and numerical parameters matching the original MATLAB file."""

    S0: float = 1.0       # Initial stock price
    K: float = 1.0        # Strike price
    r: float = 0.0        # Risk-free rate
    z: float = 0.4        # Initial variance/volatility input used in the MATLAB code
    alpha: float = 0.6    # Fractional roughness parameter
    lam: float = 2.0      # Mean-reversion speed lambda
    theta: float = 0.04   # Long-term variance level
    rho: float = -0.5     # Correlation
    nu: float = 0.05      # Vol-of-vol parameter used by the MATLAB code
    t: float = 1.0        # Maturity

    R: float = 1.5        # Fourier damping parameter
    u_lower: float = 0.0
    u_upper: float = 25.0


@dataclass(frozen=True)
class RoughHestonModel:
    """Rough Heston option pricing model.

    Parameters are fixed when the model is created. Pricing is performed with
    calculate(), which can choose between the quadratic-implicit solver and the
    original explicit predictor-corrector baseline.
    """

    params: RoughHestonParams = RoughHestonParams()

    def calculate(
        self,
        NOuter: int,
        NInner: int,
        method: SolverMethod = "implicit",
        return_details: bool = False,
    ):
        """Price a European call option under the configured parameters."""
        return _calculate_price(
            NOuter=NOuter,
            NInner=NInner,
            params=self.params,
            method=method,
            return_details=return_details,
        )


# ============================================================
# Quadrature helpers
# ============================================================

def gauss_legendre(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature nodes and weights on [a, b]."""
    if n <= 0:
        raise ValueError("n must be positive.")

    x_std, w_std = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * x_std + 0.5 * (a + b)
    w = 0.5 * (b - a) * w_std
    return x, w


def composite_simpson(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Composite Simpson rule nodes and weights on [a, b]."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n % 2 != 0:
        raise ValueError("Composite Simpson requires even n. Please use an even NInner.")

    x = np.linspace(a, b, n + 1)
    h = (b - a) / n

    w = np.ones(n + 1, dtype=float)
    w[1::2] *= 4.0
    w[2:n:2] *= 2.0
    w *= h / 3.0

    return x, w


def fgc(K: float, u: np.ndarray) -> np.ndarray:
    """Lewis-style Fourier transform of the call payoff.

    Same formula as MATLAB Fgc:
        -K^(i*u + 1) / (u^2 - i*u)
    """
    return -(K ** (1j * u + 1.0)) / (u**2 - 1j * u)


# ============================================================
# Rough Heston Riccati structure
# ============================================================

def riccati_coefficients(
    uL: np.ndarray,
    lam: float,
    rho: float,
    nu: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return A, B, C such that F(h) = A + B*h + C*h^2.

    The MATLAB code uses
        F(h) = 0.5 * [ -uL^2 - i*uL
                       + 2*lambda*(i*rho*nu*uL - 1)*h
                       + (lambda*nu*h)^2 ]

    Therefore:
        A = 0.5*(-uL^2 - i*uL)
        B = lambda*(i*rho*nu*uL - 1)
        C = 0.5*(lambda*nu)^2
    """
    A = 0.5 * (-uL**2 - 1j * uL)
    B = lam * (1j * rho * nu * uL - 1.0)
    C = 0.5 * (lam * nu) ** 2
    return A, B, C


def quadratic_implicit_corrector(
    G: np.ndarray,
    predictor: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: float,
    a_endpoint: float,
    eps: float = 1e-14,
) -> np.ndarray:
    """Closed-form implicit Adams corrector.

    The implicit corrector solves
        h = G + a_endpoint * F(h),
    where
        F(h) = A + B*h + C*h^2.

    This becomes the complex quadratic equation
        a_endpoint*C*h^2 + (a_endpoint*B - 1)*h + (G + a_endpoint*A) = 0.

    We compute both roots and choose the one closest to the predictor value.
    This root-selection rule keeps the numerical branch continuous.
    """
    q2 = a_endpoint * C
    q1 = a_endpoint * B - 1.0
    q0 = G + a_endpoint * A

    # If the quadratic term is nearly zero, fall back to the linear equation
    # q1*h + q0 = 0.
    linear_root = -q0 / q1

    if abs(q2) < eps:
        return linear_root

    discriminant = q1**2 - 4.0 * q2 * q0
    sqrt_discriminant = np.sqrt(discriminant)

    root_plus = (-q1 + sqrt_discriminant) / (2.0 * q2)
    root_minus = (-q1 - sqrt_discriminant) / (2.0 * q2)

    use_plus = np.abs(root_plus - predictor) <= np.abs(root_minus - predictor)
    corrector = np.where(use_plus, root_plus, root_minus)
    return corrector


def _calculate_price(
    NOuter: int,
    NInner: int,
    params: RoughHestonParams,
    method: SolverMethod,
    return_details: bool = False,
):
    """Price a European call under rough Heston.

    The implicit method solves h = G + a*F(h) exactly as a
    quadratic equation at each time step and Fourier node. The explicit method
    keeps the original predictor-corrector endpoint update h = G + a*F(h^P).

    Parameters
    ----------
    NOuter : int
        Number of Gauss-Legendre nodes for Fourier integration.
    NInner : int
        Number of time steps for fractional Adams recursion. Must be even
        because Simpson quadrature is used at the end.
    params : RoughHestonParams
        Model and numerical parameters.
    method : {"implicit", "explicit"}
        Corrector method used in the fractional Adams recursion.
    return_details : bool
        If True, return (price, details_dict). Otherwise return price.

    Returns
    -------
    price : float
        Option price.
    details : dict, optional
        Returned only when return_details=True. Contains u, h, numF, L.
    """
    if NOuter <= 0:
        raise ValueError("NOuter must be positive.")
    if NInner <= 0:
        raise ValueError("NInner must be positive.")
    if NInner % 2 != 0:
        raise ValueError("NInner must be even because composite Simpson is used.")

    if method not in ("implicit", "explicit"):
        raise ValueError("method must be 'implicit' or 'explicit'.")

    p = params

    # Fourier quadrature grid.
    u, w = gauss_legendre(NOuter, p.u_lower, p.u_upper)
    uL = u.astype(np.complex128) - 1j * p.R

    # F(h) = A + B*h + C*h^2.
    A, B, C = riccati_coefficients(uL, p.lam, p.rho, p.nu)

    def F(h: np.ndarray) -> np.ndarray:
        return A + B * h + C * h**2

    # Store h(t_k, u_i) and F(h(t_k, u_i)).
    h = np.zeros((NOuter, NInner + 1), dtype=np.complex128)
    numF = np.zeros_like(h)
    h[:, 0] = 0.0
    numF[:, 0] = F(h[:, 0])
    F0 = numF[:, 0]

    delta = p.t / NInner

    # Fractional Adams weights.
    a_constant = delta**p.alpha / gamma(p.alpha + 2.0)
    b_constant = delta**p.alpha / gamma(p.alpha + 1.0)

    # Precompute a0(k), k=0,...,NInner-1.
    k_arr = np.arange(NInner, dtype=float)
    a0_values = a_constant * (
        k_arr ** (p.alpha + 1.0) - (k_arr - p.alpha) * (k_arr + 1.0) ** p.alpha
    )

    # Precompute a_mid following the same reverse-order logic as MATLAB code.
    a_all = np.arange(NInner, -1, -1, dtype=float) ** (p.alpha + 1.0)
    a_diff_order1 = a_all[:NInner] - a_all[1 : NInner + 1]
    a_diff_order2 = a_diff_order1[: NInner - 1] - a_diff_order1[1:NInner]
    a_mid = a_constant * a_diff_order2

    # Precompute predictor weights b.
    b_all = np.arange(NInner, -1, -1, dtype=float) ** p.alpha
    b_diff = b_all[:NInner] - b_all[1 : NInner + 1]
    b_coef = b_constant * b_diff

    for k in range(NInner):
        # Predictor:
        #   h_{k+1}^P = sum_{j=0}^k b_{k-j} F(h_j)
        predictor_weights = b_coef[-(k + 1) :]
        predictor = numF[:, : k + 1] @ predictor_weights

        # Historical part of Adams-Moulton corrector:
        #   G_k = a0(k)*F(h_0) + sum_{j=1}^k a_{k,j} F(h_j)
        if k == 0:
            G = a0_values[k] * F0
        else:
            corrector_history_weights = a_mid[-k:]
            G = a0_values[k] * F0 + numF[:, 1 : k + 1] @ corrector_history_weights

        if method == "implicit":
            # New method: solve h = G + a*F(h) exactly as a quadratic equation.
            corrector = quadratic_implicit_corrector(
                G=G,
                predictor=predictor,
                A=A,
                B=B,
                C=C,
                a_endpoint=a_constant,
            )
        else:
            corrector = G + a_constant * F(predictor)

        h[:, k + 1] = corrector
        numF[:, k + 1] = F(corrector)

    # Time integral in the characteristic function formula.
    _, w_inner = composite_simpson(NInner, 0.0, p.t)
    L = np.exp(p.theta * p.lam * (h @ w_inner) + p.z * (numF @ w_inner))

    # Fourier inversion.
    payoff_transform = p.S0 * fgc(p.K / p.S0, 1j * p.R - u)
    price = 2.0 * np.exp(-p.r * p.t) / (2.0 * np.pi) * np.sum(
        w * np.real(L * payoff_transform)
    )
    price = float(np.real(price))

    if return_details:
        details = {
            "u": u,
            "weights_fourier": w,
            "uL": uL,
            "h": h,
            "numF": numF,
            "L": L,
            "params": p,
            "method": method,
        }
        return price, details

    return price
