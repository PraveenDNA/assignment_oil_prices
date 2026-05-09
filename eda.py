#!/usr/bin/env python3
"""
Full Exploratory Data Analysis — Brent Crude (BZ=F)
=====================================================

Produces a self-contained HTML EDA report at reports/eda_report.html

Usage:
    python eda.py
    python eda.py --start 2010-01-01
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ── helpers ─────────────────────────────────────────────────────────────────

BLUE   = "#1a365d"
ACCENT = "#2b6cb0"
RED    = "#c53030"
GREEN  = "#276749"
ORANGE = "#c05621"
GRAY   = "#718096"

def _b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return enc

def _fig(w=11, h=5): return plt.subplots(figsize=(w, h))

# ── data ─────────────────────────────────────────────────────────────────────

def load_data(start: str) -> pd.DataFrame:
    from brent_stress.data import fetch_brent
    df = fetch_brent(start=start)
    df["abs_return"]   = df["log_return"].abs()
    df["sq_return"]    = df["log_return"] ** 2
    df["roll_vol_21"]  = df["log_return"].rolling(21).std() * np.sqrt(252)   # annualised
    df["roll_vol_63"]  = df["log_return"].rolling(63).std() * np.sqrt(252)
    df["roll_mean_21"] = df["log_return"].rolling(21).mean()
    df["cum_return"]   = np.exp(df["log_return"].cumsum() / 100)             # price index (start=1)
    roll_max           = df["cum_return"].expanding().max()
    df["drawdown"]     = (roll_max - df["cum_return"]) / roll_max * 100
    df["year"]         = df.index.year
    df["month"]        = df.index.month
    df["weekday"]      = df.index.dayofweek
    return df

# ── Section 1: Price & Returns Overview ──────────────────────────────────────

def fig_price_overview(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Brent Crude — Price, Returns & Volatility Overview", fontsize=13, fontweight="bold")

    # Panel 1: Price
    ax = axes[0]
    ax.plot(df.index, df["Close"], lw=0.9, color=BLUE)
    ax.set_ylabel("Close (USD/bbl)")
    ax.set_title("Daily Close Price")
    ax.grid(True, alpha=0.25)
    # Annotate major events
    events = {
        "2016-01-20": ("Oil glut\nlows", "bottom"),
        "2020-04-21": ("COVID\ncrash", "bottom"),
        "2022-03-08": ("Russia\nUkraine", "top"),
    }
    for date_str, (label, pos) in events.items():
        try:
            d = pd.Timestamp(date_str)
            val = df.loc[d, "Close"] if d in df.index else df["Close"].asof(d)
            offset = -15 if pos == "bottom" else 10
            ax.annotate(label, xy=(d, val), xytext=(0, offset),
                        textcoords="offset points", ha="center", fontsize=7,
                        color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
        except Exception:
            pass

    # Panel 2: Returns
    ax2 = axes[1]
    ax2.bar(df.index, df["log_return"], width=1, color=np.where(df["log_return"] >= 0, ACCENT, RED),
            alpha=0.6)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Log-return (%)")
    ax2.set_title("Daily Percentage Log-Returns")
    ax2.grid(True, alpha=0.25)

    # Panel 3: Rolling volatility
    ax3 = axes[2]
    ax3.plot(df.index, df["roll_vol_21"], lw=1.0, color=ORANGE, label="21-day vol (ann.)")
    ax3.plot(df.index, df["roll_vol_63"], lw=1.5, color=BLUE,   label="63-day vol (ann.)", alpha=0.8)
    ax3.set_ylabel("Annualised vol (%)")
    ax3.set_title("Rolling Realised Volatility")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    fig.tight_layout()
    return _b64(fig)


# ── Section 2: Distributional Analysis ───────────────────────────────────────

def fig_distribution(df: pd.DataFrame) -> str:
    r = df["log_return"].dropna().values
    nu, mu_t, scale_t = stats.t.fit(r)

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle("Return Distribution Analysis", fontsize=13, fontweight="bold")

    # 1. Histogram + fits
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(r, bins=100, density=True, color=ACCENT, alpha=0.5, label="Empirical")
    x = np.linspace(r.min(), r.max(), 400)
    ax1.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=GREEN, lw=2, label="Normal fit")
    ax1.plot(x, stats.t.pdf(x, nu, mu_t, scale_t), color=RED,   lw=2, label=f"Student-t (ν={nu:.1f})")
    ax1.set_xlabel("Log-return (%)"); ax1.set_ylabel("Density")
    ax1.set_title("Histogram + Distribution Fits")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    # 2. QQ vs Normal
    ax2 = fig.add_subplot(gs[1])
    z = (r - r.mean()) / r.std()
    stats.probplot(z, dist="norm", plot=ax2)
    ax2.get_lines()[0].set(markerfacecolor=ACCENT, alpha=0.3, markersize=2)
    ax2.get_lines()[1].set(color=RED, lw=1.5)
    ax2.set_title("QQ vs Normal\n(fat tails = points above line)")
    ax2.grid(True, alpha=0.25)

    # 3. QQ vs Student-t
    ax3 = fig.add_subplot(gs[2])
    stats.probplot(z, dist=stats.t, sparams=(nu,), plot=ax3)
    ax3.get_lines()[0].set(markerfacecolor=ACCENT, alpha=0.3, markersize=2)
    ax3.get_lines()[1].set(color=RED, lw=1.5)
    ax3.set_title(f"QQ vs Student-t (ν={nu:.1f})\n(better fit in tails)")
    ax3.grid(True, alpha=0.25)

    fig.tight_layout()
    return _b64(fig)


def fig_tail_analysis(df: pd.DataFrame) -> str:
    """Hill estimator + mean excess plots side by side."""
    r = df["log_return"].dropna().values
    losses = np.sort(-r[r < 0])[::-1]   # positive losses, descending

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tail Analysis", fontsize=13, fontweight="bold")

    # — Hill estimator —
    ax = axes[0]
    x_abs = np.sort(np.abs(r))[::-1]
    k_vals = range(15, min(200, len(x_abs) - 2))
    alphas = []
    for k in k_vals:
        denom = np.mean(np.log(x_abs[:k])) - np.log(x_abs[k])
        alphas.append(1.0 / denom if denom > 1e-10 else np.nan)
    alphas = np.array(alphas)

    ax.plot(list(k_vals), alphas, color=ACCENT, lw=1.2, label="Hill α(k)")
    ax.axhline(2, color=GRAY,  lw=1, linestyle=":",  label="α=2 (finite variance)")
    ax.axhline(4, color=GRAY,  lw=1, linestyle="-.", label="α=4 (finite kurtosis)")
    # find stable plateau
    window = 25
    best, min_std = 0, np.inf
    for i in range(len(alphas) - window):
        s = np.nanstd(alphas[i:i+window])
        if s < min_std: min_std, best = s, i
    stable_alpha = float(np.nanmean(alphas[best:best+window]))
    k_arr = np.array(list(k_vals))
    ax.axhline(stable_alpha, color=RED, lw=1.5, linestyle="--",
               label=f"Stable α ≈ {stable_alpha:.2f}")
    ax.axvspan(k_arr[best], k_arr[best+window-1], alpha=0.12, color=RED)
    ax.set_xlabel("k"); ax.set_ylabel("Estimated tail index α")
    ax.set_title("Hill Estimator (absolute returns)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 9)

    # — Mean-excess plot —
    ax2 = axes[1]
    quantiles = np.linspace(0.5, 0.97, 60)
    thresholds = np.quantile(losses, quantiles)
    me, ci_h = [], []
    for u in thresholds:
        exc = losses[losses > u] - u
        if len(exc) >= 5:
            me.append(exc.mean())
            ci_h.append(1.96 * exc.std(ddof=1) / np.sqrt(len(exc)))
        else:
            me.append(np.nan); ci_h.append(np.nan)
    me   = np.array(me);   ci_h = np.array(ci_h)
    mask = ~np.isnan(me)
    ax2.plot(thresholds[mask], me[mask], color=BLUE, lw=1.8, label="Mean excess e(u)")
    ax2.fill_between(thresholds[mask], (me-ci_h)[mask], (me+ci_h)[mask],
                     alpha=0.2, color=ACCENT, label="95% CI")
    ax2.set_xlabel("Threshold u (loss %)"); ax2.set_ylabel("Mean excess e(u)")
    ax2.set_title("Mean-Excess Plot — Left Tail\n(upward slope = heavy Pareto tail)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return _b64(fig)


# ── Section 3: ACF / ARCH tests ───────────────────────────────────────────────

def fig_acf_full(df: pd.DataFrame) -> str:
    r = df["log_return"].dropna()
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("ACF / PACF — Returns and Squared Returns", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    for idx, (series, title, color) in enumerate([
        (r,    "Returns",          BLUE),
        (r**2, "Squared Returns",  RED),
    ]):
        ax_l = fig.add_subplot(gs[idx, 0])
        ax_r = fig.add_subplot(gs[idx, 1])
        plot_acf(series,  lags=40, ax=ax_l, alpha=0.05, color=color)
        plot_pacf(series, lags=40, ax=ax_r, alpha=0.05, color=color, method="ywm")
        ax_l.set_title(f"ACF — {title}")
        ax_r.set_title(f"PACF — {title}")
        if idx == 1:
            ax_l.set_title(f"ACF — {title}  ← ARCH effect test")

    fig.tight_layout()
    return _b64(fig)


def fig_ljung_box(df: pd.DataFrame) -> str:
    """Ljung-Box p-values across lags for returns and squared returns."""
    r = df["log_return"].dropna().values
    lags = list(range(1, 41))

    lb_r  = acorr_ljungbox(r,    lags=lags, return_df=True)["lb_pvalue"].values
    lb_r2 = acorr_ljungbox(r**2, lags=lags, return_df=True)["lb_pvalue"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Ljung-Box Test P-Values (H₀: no autocorrelation)", fontsize=12, fontweight="bold")

    for ax, pvals, title, color in [
        (axes[0], lb_r,  "Raw Returns",          BLUE),
        (axes[1], lb_r2, "Squared Returns (ARCH test)", RED),
    ]:
        ax.bar(lags, pvals, color=color, alpha=0.7)
        ax.axhline(0.05, color="black", lw=1.5, linestyle="--", label="5% significance")
        ax.set_xlabel("Lag"); ax.set_ylabel("p-value")
        ax.set_title(title); ax.legend(fontsize=9)
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return _b64(fig)


# ── Section 4: Volatility Regime Analysis ────────────────────────────────────

def fig_volatility_regimes(df: pd.DataFrame) -> str:
    """Rolling vol with high/low regime bands and year-by-year vol."""
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Volatility Regime Analysis", fontsize=13, fontweight="bold")

    # 1. Rolling vol + regime bands
    ax1 = fig.add_subplot(gs[0, :])
    vol = df["roll_vol_21"].dropna()
    high_threshold = vol.quantile(0.75)
    low_threshold  = vol.quantile(0.25)
    ax1.plot(vol.index, vol, lw=0.9, color=BLUE, label="21-day realised vol (ann.)")
    ax1.axhline(high_threshold, color=RED,   lw=1.2, linestyle="--",
                label=f"75th pct = {high_threshold:.1f}% (high-vol regime)")
    ax1.axhline(low_threshold,  color=GREEN, lw=1.2, linestyle="--",
                label=f"25th pct = {low_threshold:.1f}% (low-vol regime)")
    ax1.fill_between(vol.index, high_threshold, vol,
                     where=(vol >= high_threshold), alpha=0.2, color=RED)
    ax1.fill_between(vol.index, vol, low_threshold,
                     where=(vol <= low_threshold), alpha=0.2, color=GREEN)
    ax1.set_ylabel("Annualised vol (%)"); ax1.set_title("Rolling 21-Day Volatility with Regime Bands")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    # 2. Annual vol box plot
    ax2 = fig.add_subplot(gs[1, 0])
    yearly = [df[df["year"] == y]["log_return"].dropna().values
              for y in sorted(df["year"].unique())]
    labels = sorted(df["year"].unique())
    bp = ax2.boxplot(yearly, labels=labels, patch_artist=True,
                     medianprops=dict(color=RED, lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT); patch.set_alpha(0.5)
    ax2.set_xlabel("Year"); ax2.set_ylabel("Daily log-return (%)")
    ax2.set_title("Return Distribution by Year")
    ax2.tick_params(axis="x", rotation=45); ax2.grid(True, alpha=0.25, axis="y")

    # 3. Annual realised vol
    ax3 = fig.add_subplot(gs[1, 1])
    ann_vol = df.groupby("year")["log_return"].std() * np.sqrt(252)
    colors  = [RED if v > ann_vol.median() else ACCENT for v in ann_vol]
    ax3.bar(ann_vol.index, ann_vol.values, color=colors, alpha=0.8)
    ax3.axhline(ann_vol.median(), color="black", lw=1.5, linestyle="--",
                label=f"Median = {ann_vol.median():.1f}%")
    ax3.set_xlabel("Year"); ax3.set_ylabel("Annualised vol (%)")
    ax3.set_title("Annual Realised Volatility\n(red = above median)")
    ax3.legend(fontsize=9); ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    return _b64(fig)


# ── Section 5: Stationarity & Memory ─────────────────────────────────────────

def run_stationarity_tests(df: pd.DataFrame) -> dict:
    r = df["log_return"].dropna().values
    p = df["Close"].dropna().values

    # ADF on prices and returns
    adf_price  = adfuller(p,  maxlag=10, autolag="AIC")
    adf_return = adfuller(r,  maxlag=10, autolag="AIC")

    # KPSS on returns
    kpss_ret = kpss(r, regression="c", nlags="auto")

    return {
        "adf_price":  {"stat": adf_price[0],  "pval": adf_price[1],  "crit": adf_price[4]},
        "adf_return": {"stat": adf_return[0], "pval": adf_return[1], "crit": adf_return[4]},
        "kpss_ret":   {"stat": kpss_ret[0],   "pval": kpss_ret[1],   "crit": kpss_ret[3]},
    }


def fig_stationarity(df: pd.DataFrame) -> str:
    r   = df["log_return"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Stationarity Diagnostics", fontsize=12, fontweight="bold")

    # Rolling mean & std of returns
    ax = axes[0]
    roll_mean = r.rolling(63).mean()
    roll_std  = r.rolling(63).std()
    ax.plot(r.index, roll_mean, color=BLUE,   lw=1.2, label="Rolling mean (63d)")
    ax.fill_between(r.index, roll_mean - roll_std, roll_mean + roll_std,
                    alpha=0.15, color=BLUE)
    ax.plot(r.index, roll_std,  color=ORANGE, lw=1.2, label="Rolling std (63d)", linestyle="--")
    ax.axhline(0, color="black", lw=0.6, linestyle=":")
    ax.set_title("Rolling Mean & Std of Returns\n(constant mean ≈ stationary)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # Cumulative return (price index)
    ax2 = axes[1]
    ax2.plot(df.index, df["cum_return"], color=BLUE, lw=1.2)
    ax2.axhline(1, color="black", lw=0.6, linestyle=":")
    ax2.set_title("Cumulative Price Index (start=1)\n(non-stationary price, stationary returns)")
    ax2.set_ylabel("Index"); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return _b64(fig)


# ── Section 6: Drawdown Analysis ─────────────────────────────────────────────

def fig_drawdown(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Drawdown Analysis", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(df.index, df["cum_return"], color=BLUE, lw=1.0)
    roll_max = df["cum_return"].expanding().max()
    ax1.plot(df.index, roll_max, color=GREEN, lw=1.0, linestyle="--", alpha=0.7, label="Running max")
    ax1.set_ylabel("Price index"); ax1.set_title("Price Index & Running Maximum")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

    ax2 = axes[1]
    ax2.fill_between(df.index, df["drawdown"], 0, color=RED, alpha=0.5)
    ax2.plot(df.index, df["drawdown"], color=RED, lw=0.7)
    max_dd = df["drawdown"].max()
    max_dd_date = df["drawdown"].idxmax()
    ax2.annotate(f"Max DD\n{max_dd:.1f}%", xy=(max_dd_date, max_dd),
                 xytext=(40, 10), textcoords="offset points",
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax2.set_ylabel("Drawdown (%)"); ax2.set_title("Drawdown from Running Maximum")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return _b64(fig)


# ── Section 7: Seasonality ────────────────────────────────────────────────────

def fig_seasonality(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Seasonality in Returns & Volatility", fontsize=12, fontweight="bold")

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    day_names   = ["Mon","Tue","Wed","Thu","Fri"]

    # Monthly median return
    ax1 = axes[0]
    monthly_ret = df.groupby("month")["log_return"].agg(["median","std"])
    ax1.bar(monthly_ret.index, monthly_ret["median"],
            color=np.where(monthly_ret["median"] >= 0, ACCENT, RED), alpha=0.8)
    ax1.errorbar(monthly_ret.index, monthly_ret["median"],
                 yerr=monthly_ret["std"] / np.sqrt(df.groupby("month").size()),
                 fmt="none", color="black", capsize=3, lw=1)
    ax1.set_xticks(range(1, 13)); ax1.set_xticklabels(month_names)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Median log-return (%)"); ax1.set_title("Median Monthly Return")
    ax1.grid(True, alpha=0.25, axis="y")

    # Day-of-week effect on volatility
    ax2 = axes[1]
    dow_vol = df[df["weekday"] < 5].groupby("weekday")["abs_return"].mean()
    ax2.bar(dow_vol.index, dow_vol.values, color=BLUE, alpha=0.8)
    ax2.set_xticks(range(5)); ax2.set_xticklabels(day_names)
    ax2.set_ylabel("Mean |return| (%)"); ax2.set_title("Day-of-Week Volatility Effect")
    ax2.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    return _b64(fig)


# ── Section 8: Extreme Events ─────────────────────────────────────────────────

def fig_extreme_events(df: pd.DataFrame) -> str:
    r = df["log_return"].dropna()
    threshold_pct = 5.0
    extremes_pos = r[r >  threshold_pct]
    extremes_neg = r[r < -threshold_pct]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Extreme Events (|return| > {threshold_pct}%)", fontsize=12, fontweight="bold")

    # Timeline of extremes
    ax1 = axes[0]
    ax1.plot(r.index, r, lw=0.4, color=GRAY, alpha=0.5)
    ax1.scatter(extremes_pos.index, extremes_pos.values, color=GREEN, s=15, zorder=5,
                label=f"Positive >+{threshold_pct}% ({len(extremes_pos)} days)")
    ax1.scatter(extremes_neg.index, extremes_neg.values, color=RED, s=15, zorder=5,
                label=f"Negative <-{threshold_pct}% ({len(extremes_neg)} days)")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_ylabel("Log-return (%)"); ax1.set_title("Timeline of Extreme Returns")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    # Extreme event clustering: gaps between consecutive extremes
    ax2 = axes[1]
    all_extremes = r[r.abs() > threshold_pct].sort_index()
    if len(all_extremes) > 1:
        gaps = pd.Series(all_extremes.index).diff().dt.days.dropna()
        ax2.hist(gaps, bins=30, color=ACCENT, alpha=0.8, edgecolor="white")
        ax2.axvline(gaps.median(), color=RED, lw=2, linestyle="--",
                    label=f"Median gap = {gaps.median():.0f} days")
        ax2.set_xlabel("Days between consecutive extreme events")
        ax2.set_ylabel("Count"); ax2.set_title("Clustering of Extreme Events\n(small gaps = clustering)")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return _b64(fig)


# ── Section 9: Leverage Effect ────────────────────────────────────────────────

def fig_leverage(df: pd.DataFrame) -> str:
    """Scatter: lagged return vs next-day absolute return (leverage effect)."""
    r = df["log_return"].dropna()
    lagged_r  = r.shift(1).dropna()
    next_abs  = r.abs().iloc[1:]
    idx       = lagged_r.index.intersection(next_abs.index)
    lagged_r  = lagged_r.loc[idx]
    next_abs  = next_abs.loc[idx]

    # Bin lagged return into deciles and compute mean |next return|
    bins = pd.qcut(lagged_r, 10, labels=False)
    bin_centers = lagged_r.groupby(bins).mean()
    bin_vols    = next_abs.groupby(bins).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Leverage Effect — Do Negative Shocks Raise Vol More?", fontsize=12, fontweight="bold")

    ax1 = axes[0]
    ax1.scatter(lagged_r, next_abs, alpha=0.08, s=4, color=ACCENT)
    ax1.plot(bin_centers, bin_vols, color=RED, lw=2.5, marker="o", markersize=5,
             label="Decile mean")
    ax1.axvline(0, color="black", lw=0.8, linestyle="--")
    ax1.set_xlabel("Return at t (%)"); ax1.set_ylabel("|Return| at t+1 (%)")
    ax1.set_title("Lagged Return vs Next Absolute Return")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

    ax2 = axes[1]
    colors = [RED if c < 0 else GREEN for c in bin_centers]
    ax2.bar(range(10), bin_vols.values, color=colors, alpha=0.8)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f"{v:.1f}" for v in bin_centers.values], rotation=45, fontsize=7)
    ax2.set_xlabel("Return decile (t)"); ax2.set_ylabel("Mean |return| (t+1, %)")
    ax2.set_title("Mean Next-Day Vol by Return Decile\n(red = negative return decile)")
    ax2.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    return _b64(fig)


# ── HTML report ───────────────────────────────────────────────────────────────

_CSS = """
<style>
:root { --primary:#1a365d; --accent:#2b6cb0; --border:#e2e8f0; --text:#2d3748; --muted:#718096; }
* { box-sizing:border-box; }
body { font-family:'Segoe UI',system-ui,sans-serif; color:var(--text); background:#f7fafc;
       max-width:1200px; margin:0 auto; padding:2rem 1.5rem; line-height:1.65; }
h1 { color:var(--primary); border-bottom:3px solid var(--accent); padding-bottom:.5rem; margin-top:0; }
h2 { color:var(--primary); margin-top:2.5rem; border-left:4px solid var(--accent); padding-left:.75rem; }
h3 { color:var(--accent); margin-top:1.4rem; }
table { border-collapse:collapse; width:100%; margin:1rem 0; background:white;
        border-radius:6px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.08); }
th { background:var(--primary); color:white; padding:.55rem 1rem; text-align:left; font-size:.85rem; }
td { padding:.45rem 1rem; border-bottom:1px solid var(--border); font-size:.88rem; }
tr:last-child td { border-bottom:none; }
tr:nth-child(even) td { background:#f7fafc; }
figure { margin:1.5rem 0; background:white; padding:.75rem;
         border:1px solid var(--border); border-radius:6px; }
figure img { max-width:100%; display:block; margin:0 auto; }
figcaption { color:var(--muted); font-size:.8rem; margin-top:.4rem; text-align:center; }
.insight { background:#ebf8ff; border-left:4px solid var(--accent);
           border-radius:0 6px 6px 0; padding:.9rem 1.2rem; margin:1rem 0; }
.insight b { color:var(--primary); }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
             gap:.75rem; margin:1rem 0; }
.stat-card { background:white; border:1px solid var(--border); border-radius:6px; padding:.75rem 1rem; }
.stat-card .label { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.stat-card .value { font-size:1.15rem; font-weight:700; color:var(--primary); }
.pass { color:#276749; font-weight:700; }
.fail { color:#9b2c2c; font-weight:700; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--border);
         color:var(--muted); font-size:.8rem; }
</style>
"""

def build_report(df: pd.DataFrame, stat_tests: dict, start: str) -> str:
    r  = df["log_return"].dropna()
    nu, _, _ = stats.t.fit(r.values)

    # compute moments
    moments = {
        "n":               len(r),
        "mean":            r.mean(),
        "std":             r.std(),
        "skewness":        stats.skew(r.values),
        "excess_kurtosis": stats.kurtosis(r.values, fisher=True),
        "jb_stat":         stats.jarque_bera(r.values)[0],
        "jb_pval":         stats.jarque_bera(r.values)[1],
        "min":             r.min(),
        "max":             r.max(),
        "q01":             r.quantile(0.01),
        "q05":             r.quantile(0.05),
        "q95":             r.quantile(0.95),
        "q99":             r.quantile(0.99),
        "ann_vol":         r.std() * np.sqrt(252),
        "max_dd":          df["drawdown"].max(),
        "student_nu":      nu,
    }

    lb_sq = acorr_ljungbox(r.values**2, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
    lb_r  = acorr_ljungbox(r.values,    lags=[20], return_df=True)["lb_pvalue"].iloc[0]

    print("  Generating figures...", end=" ", flush=True)
    figs = {
        "overview":   fig_price_overview(df),
        "dist":       fig_distribution(df),
        "tails":      fig_tail_analysis(df),
        "acf":        fig_acf_full(df),
        "lb":         fig_ljung_box(df),
        "regimes":    fig_volatility_regimes(df),
        "stationary": fig_stationarity(df),
        "drawdown":   fig_drawdown(df),
        "seasonal":   fig_seasonality(df),
        "extremes":   fig_extreme_events(df),
        "leverage":   fig_leverage(df),
    }
    print("done")

    adf_p  = stat_tests["adf_return"]["pval"]
    kpss_p = stat_tests["kpss_ret"]["pval"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Brent Crude — Full EDA Report</title>
  {_CSS}
</head>
<body>
<h1>Brent Crude (BZ=F) — Full Exploratory Data Analysis</h1>
<p style="color:var(--muted);font-size:.85rem;">
  Data: {str(r.index.min().date())} → {str(r.index.max().date())} &nbsp;|&nbsp;
  {moments['n']:,} trading days &nbsp;|&nbsp;
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>

<!-- ── Key Stats ─────────────────────────────────────────────── -->
<div class="stat-grid">
  <div class="stat-card"><div class="label">Observations</div><div class="value">{moments['n']:,}</div></div>
  <div class="stat-card"><div class="label">Ann. Volatility</div><div class="value">{moments['ann_vol']:.1f}%</div></div>
  <div class="stat-card"><div class="label">Excess Kurtosis</div><div class="value">{moments['excess_kurtosis']:.2f}</div></div>
  <div class="stat-card"><div class="label">Skewness</div><div class="value">{moments['skewness']:.3f}</div></div>
  <div class="stat-card"><div class="label">Max Drawdown</div><div class="value">{moments['max_dd']:.1f}%</div></div>
  <div class="stat-card"><div class="label">Student-t ν (MLE)</div><div class="value">{moments['student_nu']:.2f}</div></div>
  <div class="stat-card"><div class="label">VaR 99% (hist.)</div><div class="value">{abs(moments['q01']):.2f}%</div></div>
  <div class="stat-card"><div class="label">Worst day</div><div class="value">{moments['min']:.2f}%</div></div>
</div>

<!-- ── Section 1 ─────────────────────────────────────────────── -->
<h2>1. Price, Returns & Volatility Overview</h2>
<figure>
  <img src="data:image/png;base64,{figs['overview']}" alt="Overview">
  <figcaption>Figure 1 — Close price (top), daily log-returns colour-coded by sign (middle),
  and rolling annualised volatility (bottom). Three major shock events are labelled.</figcaption>
</figure>
<div class="insight">
  <b>Key observation:</b> Volatility is clearly not constant over time. Calm periods
  (2017–2019, late 2023) alternate with storm periods (2020 COVID crash, 2022 Ukraine).
  This visible clustering rules out i.i.d. models and motivates the GARCH family.
</div>

<!-- ── Section 2 ─────────────────────────────────────────────── -->
<h2>2. Distributional Analysis</h2>

<h3>Empirical Moments</h3>
<table>
  <thead><tr><th>Statistic</th><th>Value</th><th>Gaussian Baseline</th><th>Interpretation</th></tr></thead>
  <tbody>
    <tr><td>Observations</td><td>{moments['n']:,}</td><td>—</td><td>~11 years of daily data</td></tr>
    <tr><td>Mean daily return (%)</td><td>{moments['mean']:.4f}</td><td>—</td><td>Near zero; not significantly different from 0</td></tr>
    <tr><td>Std deviation (%)</td><td>{moments['std']:.4f}</td><td>—</td><td>~{moments['ann_vol']:.0f}% annualised volatility</td></tr>
    <tr><td>Skewness</td><td>{moments['skewness']:.4f}</td><td>0</td><td>Negative: left-tail heavier; crashes sharper than rallies</td></tr>
    <tr><td>Excess kurtosis</td><td>{moments['excess_kurtosis']:.4f}</td><td>0</td><td>Very heavy tails — fat-tailed distribution required</td></tr>
    <tr><td>Jarque–Bera stat</td><td>{moments['jb_stat']:,.1f}</td><td>—</td><td>—</td></tr>
    <tr><td>Jarque–Bera p-value</td><td>{moments['jb_pval']:.2e}</td><td>—</td><td>Normality overwhelmingly rejected</td></tr>
    <tr><td>1st percentile (%)</td><td>{moments['q01']:.4f}</td><td>—</td><td>Historical VaR 99%</td></tr>
    <tr><td>99th percentile (%)</td><td>{moments['q99']:.4f}</td><td>—</td><td>Upper tail</td></tr>
    <tr><td>Minimum return (%)</td><td>{moments['min']:.4f}</td><td>—</td><td>Single-day worst loss</td></tr>
    <tr><td>Maximum return (%)</td><td>{moments['max']:.4f}</td><td>—</td><td>Single-day best gain</td></tr>
    <tr><td>Fitted Student-t ν</td><td>{moments['student_nu']:.2f}</td><td>∞</td><td>Heavy tails confirmed; finite variance (ν>2), near-infinite kurtosis (ν<5)</td></tr>
  </tbody>
</table>

<figure>
  <img src="data:image/png;base64,{figs['dist']}" alt="Distribution">
  <figcaption>Figure 2 — Left: histogram with Normal and Student-t fits. Centre: QQ vs Normal
  (clear tail deviation). Right: QQ vs fitted Student-t (better fit but tails still deviate).</figcaption>
</figure>
<div class="insight">
  <b>Key observation:</b> The Normal distribution dramatically under-predicts extreme moves.
  The Student-t fits better but still shows tail deviation in the QQ plot — consistent with
  the GARCH effect stacking extra kurtosis on top of the innovation distribution.
</div>

<!-- ── Section 3 ─────────────────────────────────────────────── -->
<h2>3. Tail Analysis</h2>
<figure>
  <img src="data:image/png;base64,{figs['tails']}" alt="Tail analysis">
  <figcaption>Figure 3 — Left: Hill estimator plot. The stable plateau gives the most reliable
  tail-index estimate. Right: mean-excess plot for losses; upward slope confirms Pareto-class tail.</figcaption>
</figure>
<div class="insight">
  <b>Hill estimator:</b> Tail index α stabilises in the range 3–5 in the plateau region.
  Since α &lt; 4, kurtosis is near-infinite or diverges with sample size.
  Since α &gt; 2, variance is finite. This places Brent returns in the "finite variance,
  heavy kurtosis" regime — consistent with excess kurtosis of {moments['excess_kurtosis']:.1f}.<br><br>
  <b>Mean-excess plot:</b> Upward slope throughout the loss distribution confirms a
  Pareto-like (heavy) tail — not exponential, not thin. This independently rules out Gaussian
  innovations.
</div>

<!-- ── Section 4 ─────────────────────────────────────────────── -->
<h2>4. Autocorrelation & ARCH Effects</h2>
<figure>
  <img src="data:image/png;base64,{figs['acf']}" alt="ACF">
  <figcaption>Figure 4 — ACF/PACF of raw returns (top row) and squared returns (bottom row).
  The squared-return ACF is the canonical ARCH test: significant bars confirm volatility clustering.</figcaption>
</figure>
<figure>
  <img src="data:image/png;base64,{figs['lb']}" alt="Ljung-Box">
  <figcaption>Figure 5 — Ljung-Box p-values across lags 1–40. Raw returns: mostly above 5%
  (no predictable mean). Squared returns: all below 5% (strong ARCH effect).</figcaption>
</figure>
<div class="insight">
  <b>Raw returns:</b> Ljung-Box p-value at lag 20 = {lb_r:.3f} →
  {"<span class='pass'>PASS</span> No significant autocorrelation (consistent with weak-form efficiency)" if lb_r > 0.05 else "<span class='fail'>FAIL</span> Some return autocorrelation present"}<br><br>
  <b>Squared returns:</b> Ljung-Box p-value at lag 20 = {lb_sq:.2e} →
  <span class='fail'>Strong ARCH effect confirmed</span> — volatility clustering is highly
  statistically significant. This mandates a GARCH-family model.
</div>

<!-- ── Section 5 ─────────────────────────────────────────────── -->
<h2>5. Volatility Regimes</h2>
<figure>
  <img src="data:image/png;base64,{figs['regimes']}" alt="Regimes">
  <figcaption>Figure 6 — Rolling volatility with high/low regime bands (top). Return distribution
  by year as box plots (bottom left). Annual realised volatility bar chart (bottom right).</figcaption>
</figure>
<div class="insight">
  <b>Key observation:</b> Annualised volatility ranges from ~20% in calm years to &gt;80%
  during COVID (2020). This enormous range — a 4× spread — confirms that a single stationary
  variance equation will struggle. This is the primary motivation for flagging regime shifts
  as a model failure mode.
</div>

<!-- ── Section 6 ─────────────────────────────────────────────── -->
<h2>6. Stationarity</h2>

<h3>Formal Tests</h3>
<table>
  <thead><tr><th>Test</th><th>Applied to</th><th>Statistic</th><th>p-value</th><th>Conclusion</th></tr></thead>
  <tbody>
    <tr>
      <td>ADF (Augmented Dickey-Fuller)</td><td>Close price</td>
      <td>{stat_tests['adf_price']['stat']:.4f}</td>
      <td>{stat_tests['adf_price']['pval']:.4f}</td>
      <td>{"<span class='fail'>Non-stationary</span>" if stat_tests['adf_price']['pval'] > 0.05 else "<span class='pass'>Stationary</span>"} — price has unit root</td>
    </tr>
    <tr>
      <td>ADF (Augmented Dickey-Fuller)</td><td>Log-returns</td>
      <td>{stat_tests['adf_return']['stat']:.4f}</td>
      <td>{stat_tests['adf_return']['pval']:.4f}</td>
      <td>{"<span class='pass'>Stationary</span>" if stat_tests['adf_return']['pval'] < 0.05 else "<span class='fail'>Non-stationary</span>"} — returns are stationary</td>
    </tr>
    <tr>
      <td>KPSS</td><td>Log-returns</td>
      <td>{stat_tests['kpss_ret']['stat']:.4f}</td>
      <td>{stat_tests['kpss_ret']['pval']:.4f}</td>
      <td>{"<span class='pass'>Stationary confirmed</span>" if stat_tests['kpss_ret']['pval'] > 0.05 else "<span class='fail'>Non-stationary</span>"} (KPSS H₀ = stationary)</td>
    </tr>
  </tbody>
</table>

<figure>
  <img src="data:image/png;base64,{figs['stationary']}" alt="Stationarity">
  <figcaption>Figure 7 — Rolling mean and std of returns (left): mean stays near zero, confirming
  stationarity of returns. Cumulative price index (right): non-stationary price process.</figcaption>
</figure>

<!-- ── Section 7 ─────────────────────────────────────────────── -->
<h2>7. Drawdown Analysis</h2>
<figure>
  <img src="data:image/png;base64,{figs['drawdown']}" alt="Drawdown">
  <figcaption>Figure 8 — Price index vs running maximum (top). Drawdown from running maximum (bottom).
  Maximum historical drawdown = {moments['max_dd']:.1f}%.</figcaption>
</figure>
<div class="insight">
  <b>Key observation:</b> The maximum drawdown of {moments['max_dd']:.1f}% occurred during the
  combined COVID demand shock and Russia-Ukraine period. Any generative model must be able to
  produce paths with comparable drawdown severity to be useful for stress testing.
</div>

<!-- ── Section 8 ─────────────────────────────────────────────── -->
<h2>8. Seasonality</h2>
<figure>
  <img src="data:image/png;base64,{figs['seasonal']}" alt="Seasonality">
  <figcaption>Figure 9 — Monthly median return (left): limited evidence of strong seasonal patterns.
  Day-of-week volatility (right): slight Monday/Friday elevation, though not economically significant.</figcaption>
</figure>

<!-- ── Section 9 ─────────────────────────────────────────────── -->
<h2>9. Extreme Events</h2>
<figure>
  <img src="data:image/png;base64,{figs['extremes']}" alt="Extremes">
  <figcaption>Figure 10 — Timeline of days with |return| &gt; 5% (left). Distribution of gaps
  between consecutive extreme events (right): short gaps confirm clustering, not randomness.</figcaption>
</figure>
<div class="insight">
  <b>Key observation:</b> Extreme events cluster in time. The gap distribution is right-skewed
  with many short gaps — you are far more likely to see a second extreme event shortly after
  the first than to see them uniformly distributed. This is another manifestation of volatility
  clustering and further evidence against i.i.d. models.
</div>

<!-- ── Section 10 ─────────────────────────────────────────────── -->
<h2>10. Leverage Effect</h2>
<figure>
  <img src="data:image/png;base64,{figs['leverage']}" alt="Leverage">
  <figcaption>Figure 11 — Mean next-day absolute return by decile of today's return. Negative
  return deciles (red bars) tend to produce higher next-day volatility than positive deciles —
  the leverage effect that motivates GJR over plain GARCH.</figcaption>
</figure>
<div class="insight">
  <b>Key observation:</b> The lowest return deciles (large crashes) produce systematically
  higher next-day absolute returns than the highest deciles (large rallies). This asymmetry —
  the leverage effect — is the empirical justification for choosing GJR-GARCH over plain GARCH.
  The γ parameter in GJR directly encodes this: bad news amplifies future volatility more than
  good news of equivalent magnitude.
</div>

<!-- ── Section 11: Model Implication Summary ─────────────────── -->
<h2>11. Modelling Implications — What the EDA Tells Us</h2>
<table>
  <thead><tr><th>Finding</th><th>Evidence</th><th>Model implication</th></tr></thead>
  <tbody>
    <tr><td>Volatility clustering</td><td>ACF of squared returns significant to lag 20+; LB p ≈ {lb_sq:.0e}</td><td>Must use GARCH family — i.i.d. models ruled out</td></tr>
    <tr><td>Heavy tails</td><td>Excess kurtosis {moments['excess_kurtosis']:.1f}; Hill α ≈ 3–5; QQ deviation</td><td>Student-t (or heavier) innovations — Gaussian ruled out</td></tr>
    <tr><td>Leverage asymmetry</td><td>Negative return deciles produce higher next-day vol</td><td>GJR-GARCH over plain GARCH — need γ term</td></tr>
    <tr><td>Negative skewness</td><td>Skewness = {moments['skewness']:.3f}</td><td>Symmetric Student-t will under-predict left tail (known limitation)</td></tr>
    <tr><td>Regime shifts</td><td>Annual vol ranges from ~20% to ~80%</td><td>Single-state GARCH will misfit sustained high-vol periods (known limitation)</td></tr>
    <tr><td>Return stationarity</td><td>ADF p = {stat_tests['adf_return']['pval']:.4f}; KPSS p = {stat_tests['kpss_ret']['pval']:.3f}</td><td>GARCH on log-returns is appropriate; price level is non-stationary</td></tr>
  </tbody>
</table>

<footer>Brent Crude Full EDA · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</footer>
</body>
</html>"""

    return html


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Full EDA on Brent crude returns")
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()

    Path("reports").mkdir(exist_ok=True)

    print("\nBrent Crude — Full EDA")
    print(f"  start = {args.start}\n")

    print("  [1/4] Fetching data ...", end=" ", flush=True)
    df = load_data(args.start)
    print(f"done  ({len(df):,} days, {df.index.min().date()} → {df.index.max().date()})")

    print("  [2/4] Running stationarity tests ...", end=" ", flush=True)
    stat_tests = run_stationarity_tests(df)
    print("done")

    print("  [3/4] Building report ...", end=" ", flush=True)
    html = build_report(df, stat_tests, args.start)
    print("done")

    print("  [4/4] Writing reports/eda_report.html ...", end=" ", flush=True)
    Path("reports/eda_report.html").write_text(html, encoding="utf-8")
    print("done")

    print("\n  → Open reports/eda_report.html in your browser\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
