"""
GJR-GARCH(1,1) with Student-t innovations.

Model equations
---------------
    r_t  = μ + ε_t
    ε_t  = σ_t · z_t,   z_t ~ t(ν)
    σ²_t = ω + (α + γ · 1_{ε_{t-1}<0}) · ε²_{t-1} + β · σ²_{t-1}

Why GJR-GARCH over plain GARCH
-------------------------------
Plain GARCH(1,1) treats positive and negative shocks symmetrically.
Brent crude is acutely sensitive to geopolitical supply disruptions: a
sudden negative shock (demand collapse, OPEC cut reversal) tends to raise
future realised volatility more than an equivalent positive shock.  The
GJR (Glosten-Jagannathan-Runkle) extension adds the asymmetric term γ,
which is statistically significant in oil markets and materially improves
tail fit.

Why Student-t over Gaussian innovations
-----------------------------------------
The Hill estimator on the historical return series typically places the
tail index α in [3, 5], implying finite variance but diverging fourth
moment.  A Gaussian innovation distribution would dramatically under-weight
extreme scenario probability.  The Student-t with estimated degrees of
freedom ν directly parameterises the tail heaviness and collapses to a
Gaussian as ν → ∞.

What the model reproduces
--------------------------
  ✓ Volatility clustering (ACF of σ²_t / squared returns)
  ✓ Leverage effect (γ > 0: negative shocks → larger subsequent σ_t)
  ✓ Heavy tails (Student-t innovations, ν typically 5–12 for Brent)

What the model does NOT reproduce
-----------------------------------
  ✗ Macro volatility regime shifts (COVID 2020, Russia 2022): a single
    stationary variance equation cannot track persistent level shifts.
    A Markov-switching GARCH would be required.
  ✗ Skewed innovations: Student-t is symmetric; Brent returns exhibit
    modest negative skewness from supply shock asymmetry.
  ✗ Cross-asset co-movement: each path is simulated independently.
  ✗ Long-memory volatility (FIGARCH territory).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

SEED: int = 42


def fit_gjr_garch(returns: pd.Series, seed: int = SEED) -> ARCHModelResult:
    """
    Fit GJR-GARCH(1,1) with Student-t innovations to percentage log-returns.

    Parameters
    ----------
    returns : pd.Series
        Percentage log-returns (i.e. 100 · log(P_t/P_{t-1})).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ARCHModelResult
        Fitted result object from the `arch` package.
        Key params: mu, omega, alpha[1], gamma[1], beta[1], nu.
    """
    np.random.seed(seed)
    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,  # o=1 adds the GJR asymmetry term (γ)
        q=1,
        dist="t",
    )
    result = model.fit(disp="off", show_warning=False)
    return result


def persistence(result: ARCHModelResult) -> float:
    """
    Compute GJR-GARCH persistence: α + γ/2 + β.

    For a stationary process this must be < 1.  Values near 1 imply
    near-integrated (IGARCH-like) behaviour — slow mean-reversion of
    conditional variance.
    """
    p = result.params
    alpha = p.get("alpha[1]", 0.0)
    gamma = p.get("gamma[1]", 0.0)
    beta = p.get("beta[1]", 0.0)
    return float(alpha + 0.5 * gamma + beta)


def simulate(
    result: ARCHModelResult,
    n_paths: int = 1000,
    horizon: int = 252,
    seed: int = SEED,
) -> np.ndarray:
    """
    Simulate `n_paths` independent return paths from the fitted GJR-GARCH model.

    Each path is initialised from the unconditional variance (not the current
    fitted state), which is appropriate for stress-test scenario generation
    where we want long-run distributional properties rather than short-term
    forecasts.

    Parameters
    ----------
    result : ARCHModelResult
        Fitted model from fit_gjr_garch().
    n_paths : int
        Number of independent Monte Carlo paths.
    horizon : int
        Length of each path in trading days.
    seed : int
        Base seed; each path i draws from RandomState(derived_seed_i) for
        full reproducibility across different n_paths values.

    Returns
    -------
    np.ndarray, shape (n_paths, horizon)
        Simulated percentage log-returns.
    """
    # arch >= 6.x uses a numpy.random.Generator stored in
    # result.model.distribution._generator (not the legacy np.random global).
    # We seed it directly for full reproducibility.
    # Per-path seeding: each path k gets a stable seed derived from the master
    # seed, so path k is identical regardless of n_paths.
    master_rng = np.random.default_rng(seed)
    path_seeds = master_rng.integers(0, 2**31, size=n_paths)

    paths = np.empty((n_paths, horizon), dtype=float)
    for i, s in enumerate(path_seeds):
        result.model.distribution._generator = np.random.default_rng(int(s))
        sim = result.model.simulate(
            result.params,
            nobs=horizon,
            burn=500,  # discard 500 burn-in obs to escape initial conditions
        )
        paths[i] = sim["data"].values

    return paths
