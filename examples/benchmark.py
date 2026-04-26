"""Small benchmark for rough-heston-qipc.

Run from the repository root after local installation:
    pip install -e ".[dev]"
    python examples/benchmark.py
"""

from __future__ import annotations

import time

from rough_heston_qipc import RoughHestonModel, RoughHestonParams


def main() -> None:
    grids = [(25, 100), (35, 200), (50, 500), (120, 1000)]
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

    print("NOuter  NInner  implicit_price implicit_time explicit_price explicit_time abs_diff")
    for n_outer, n_inner in grids:
        start = time.perf_counter()
        p_new = model.price(n_outer, n_inner, method="implicit")
        t_new = time.perf_counter() - start

        start = time.perf_counter()
        p_old = model.price(n_outer, n_inner, method="explicit")
        t_old = time.perf_counter() - start

        print(
            f"{n_outer:6d} {n_inner:7d} "
            f"{p_new: .10f} {t_new:10.4f} "
            f"{p_old: .10f} {t_old:13.4f} {abs(p_new - p_old):.3e}"
        )


if __name__ == "__main__":
    main()
