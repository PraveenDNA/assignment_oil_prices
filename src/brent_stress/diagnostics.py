"""
Statistical diagnostics for Brent crude log-returns.

These functions produce the empirical evidence that motivates the choice of
GJR-GARCH with Student-t innovations.  Each function is kept pure (no side
effects) and returns a matplotlib Figure that the report module embeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf as sm_acf
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Numerical summaries
# ---------------------------------------------------------------------------

def compute_moments(returns: pd.Series) -> dict:
    """
    Compute first four sample moments plus the Jarque–Bera test.

    Returns a flat dict with keys:
        n, mean, std, skewness, excess_kurtosis, jb_stat, jb_pvalue
    """
    r = returns.dropna().values
    jb_stat, jb_pval = stats.jarque_bera(r)
    return {
        "n": len(r),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True)),  # excess (0 = Gaussian)
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pval),
    }


def hill_estimator(returns: pd.Series, k_max: int = 150) -> dict:
    """
    Hill estimator for the tail index α of |r_t|.

    H(k) = 1 / [ (1/k) Σ_{i=1}^{k} log X_(i) - log X_(k+1) ]

    where X_(1) ≥ X_(2) ≥ ... are order statistics of absolute returns.
    α > 2 implies finite variance; α > 4 implies finite kurtosis.

    Returns dict with keys: k (list), alpha (list).
    """
    x = np.sort(np.abs(returns.dropna().values))[::-1]  # descending
    n = len(x)
    k_min = 15
    k_max = min(k_max, n - 2)
    k_values = list(range(k_min, k_max + 1))

    alphas = []
    for k in k_values:
        log_mean = np.mean(np.log(x[:k]))
        log_threshold = np.log(x[k])
        denom = log_mean - log_threshold
        alphas.append(1.0 / denom if denom > 1e-10 else np.nan)

    return {"k": k_values, "alpha": alphas}


def hill_stable_region(hill_result: dict) -> tuple[float, tuple[int, int]]:
    """
    Identify the plateau in the Hill plot: the region where α is most stable.

    Uses a rolling window of 25 to find minimum standard deviation.
    Returns (mean_alpha_in_stable_region, (k_start, k_end)).
    """
    k_vals = np.array(hill_result["k"])
    alphas = np.array(hill_result["alpha"], dtype=float)
    window = 25

    best_start_idx = 0
    min_std = np.inf
    for i in range(len(k_vals) - window):
        chunk = alphas[i : i + window]
        s = np.nanstd(chunk)
        if s < min_std:
            min_std = s
            best_start_idx = i

    stable = alphas[best_start_idx : best_start_idx + window]
    mean_alpha = float(np.nanmean(stable))
    k_range = (int(k_vals[best_start_idx]), int(k_vals[best_start_idx + window - 1]))
    return mean_alpha, k_range


def acf_significant_lag(returns: pd.Series, nlags: int = 40) -> int:
    """
    Return the highest lag with statistically significant ACF of squared returns
    (Bartlett 95% confidence interval).
    """
    r2 = returns.dropna().values ** 2
    n = len(r2)
    ci = 1.96 / np.sqrt(n)
    acf_vals = sm_acf(r2, nlags=nlags, fft=True)
    sig = [lag for lag in range(1, nlags + 1) if abs(acf_vals[lag]) > ci]
    return max(sig) if sig else 0


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def plot_return_series(df: pd.DataFrame) -> plt.Figure:
    """
    Two-panel figure: (top) Brent close price, (bottom) percentage log-returns.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Brent Crude (BZ=F) — Price and Log-Returns", fontsize=13, fontweight="bold")

    ax1.plot(df.index, df["Close"], lw=0.9, color="#1a365d")
    ax1.set_ylabel("Close (USD/bbl)")
    ax1.set_title("Daily Close Price")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["log_return"], lw=0.6, color="#2b6cb0", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.5, linestyle="--")
    ax2.set_ylabel("Log-return (%)")
    ax2.set_title("Daily Percentage Log-Returns")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_acf_diagnostics(returns: pd.Series, nlags: int = 40) -> plt.Figure:
    """
    2×2 ACF/PACF panel for returns and squared returns.

    Squared-return autocorrelation is the canonical test for ARCH effects
    (volatility clustering).  Significance here motivates the GARCH family.
    """
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("ACF / PACF Diagnostics", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax_top_l = fig.add_subplot(gs[0, 0])
    ax_top_r = fig.add_subplot(gs[0, 1])
    ax_bot_l = fig.add_subplot(gs[1, 0])
    ax_bot_r = fig.add_subplot(gs[1, 1])

    r = returns.dropna()
    plot_acf(r, lags=nlags, ax=ax_top_l, alpha=0.05, color="#1a365d")
    ax_top_l.set_title("ACF — Returns")

    plot_pacf(r, lags=nlags, ax=ax_top_r, alpha=0.05, color="#1a365d", method="ywm")
    ax_top_r.set_title("PACF — Returns")

    plot_acf(r**2, lags=nlags, ax=ax_bot_l, alpha=0.05, color="#9b2c2c")
    ax_bot_l.set_title("ACF — Squared Returns  ← ARCH test")

    plot_pacf(r**2, lags=nlags, ax=ax_bot_r, alpha=0.05, color="#9b2c2c", method="ywm")
    ax_bot_r.set_title("PACF — Squared Returns")

    return fig


def plot_qq(returns: pd.Series) -> plt.Figure:
    """
    QQ plot of standardised returns against a fitted Student-t reference.

    The degrees-of-freedom parameter ν is estimated via MLE.  Deviation from
    the reference line in the tails indicates the degree of heavy-tailedness.
    """
    r = returns.dropna().values
    z = (r - np.mean(r)) / np.std(r, ddof=1)  # standardise

    # Fit Student-t df by MLE (loc and scale fixed for standardised data)
    nu, loc, scale = stats.t.fit(r)

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(z, dist=stats.t, sparams=(nu,), plot=ax)

    ax.get_lines()[0].set(markerfacecolor="#2b6cb0", alpha=0.4, markersize=3)
    ax.get_lines()[1].set(color="#c53030", lw=1.5)
    ax.set_title(f"QQ Plot — Standardised Returns vs Student-t (ν={nu:.1f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Theoretical quantiles  [t(ν)]")
    ax.set_ylabel("Sample quantiles")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_mean_excess(returns: pd.Series, n_thresholds: int = 60) -> plt.Figure:
    """
    Mean-excess (mean residual life) plot for the left tail (losses).

    e(u) = E[X − u | X > u]  where X = −r_t (losses).

    Interpretation:
      ↑ upward slope  → heavy (Pareto-like) tail
      → flat          → exponential tail
      ↓ downward      → thin (Weibull) tail

    An upward-sloping mean-excess plot in the left tail of Brent returns
    supports heavy-tailed innovation distributions.
    """
    losses = -returns.dropna().values
    losses = np.sort(losses[losses > 0])

    quantiles = np.linspace(0.50, 0.97, n_thresholds)
    thresholds = np.quantile(losses, quantiles)

    mean_exc = []
    ci_half = []
    for u in thresholds:
        exc = losses[losses > u] - u
        if len(exc) < 5:
            mean_exc.append(np.nan)
            ci_half.append(np.nan)
        else:
            mean_exc.append(float(np.mean(exc)))
            ci_half.append(float(1.96 * np.std(exc, ddof=1) / np.sqrt(len(exc))))

    mean_exc = np.array(mean_exc)
    ci_half = np.array(ci_half)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, mean_exc, color="#1a365d", lw=1.8, label="Mean excess e(u)")
    mask = ~np.isnan(ci_half)
    ax.fill_between(
        thresholds[mask],
        (mean_exc - ci_half)[mask],
        (mean_exc + ci_half)[mask],
        alpha=0.2,
        color="#2b6cb0",
        label="95% CI",
    )
    ax.set_xlabel("Threshold u  (loss = −return, %)")
    ax.set_ylabel("Mean excess e(u)")
    ax.set_title("Mean-Excess Plot — Left Tail (Losses)", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_hill(hill_result: dict) -> plt.Figure:
    """
    Hill plot: estimated tail index α versus order statistic k.

    The stable plateau (horizontal region) gives the most reliable estimate
    of α.  α > 2 → finite variance; α > 4 → finite kurtosis.
    """
    k = np.array(hill_result["k"])
    alpha = np.array(hill_result["alpha"], dtype=float)

    mean_alpha, k_range = hill_stable_region(hill_result)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k, alpha, color="#2b6cb0", lw=1.2, label="Hill α(k)")
    ax.axhline(mean_alpha, color="#c53030", lw=1.5, linestyle="--",
               label=f"Stable mean α ≈ {mean_alpha:.2f}  (k ∈ [{k_range[0]}, {k_range[1]}])")
    ax.axhline(2.0, color="gray", lw=1.0, linestyle=":", label="α = 2  (finite-variance boundary)")
    ax.axhline(4.0, color="gray", lw=1.0, linestyle="-.", label="α = 4  (finite-kurtosis boundary)")
    ax.axvspan(k_range[0], k_range[1], alpha=0.1, color="#c53030", label="Stable region")

    ax.set_xlabel("k  (number of upper order statistics)")
    ax.set_ylabel("Estimated tail index α")
    ax.set_title("Hill Estimator — Tail Index", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(8.0, float(np.nanmax(alpha)) + 0.5))
    fig.tight_layout()
    return fig
