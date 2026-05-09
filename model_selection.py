#!/usr/bin/env python3
"""
Model Selection — Brent Crude Stress-Test
==========================================

Fits 6 GARCH-family models, scores them on 4 dimensions,
selects the best, and writes reports/model_selection.html.

Models compared
---------------
  1. GARCH(1,1)  + Normal              ← baseline
  2. GARCH(1,1)  + Student-t           ← adds heavy tails
  3. GARCH(1,1)  + Skewed Student-t    ← adds asymmetric tails
  4. GJR-GARCH(1,1) + Student-t        ← adds leverage effect
  5. GJR-GARCH(1,1) + Skewed Student-t ← leverage + asymmetric tails
  6. EGARCH(1,1)   + Student-t         ← alternative asymmetric spec

Scoring dimensions
------------------
  A. In-sample fit  : AIC, BIC (lower = better)
  B. Residual adequacy : Ljung-Box on std residuals & squared std residuals
                         (p > 0.05 = PASS = no remaining structure)
  C. VaR coverage   : Kupiec POF test at 95% and 99%
                       (p > 0.05 = PASS = correct coverage)
  D. Simulation quality : |synthetic VaR/ES - historical| relative error

Usage
-----
    python model_selection.py
    python model_selection.py --start 2014-01-01
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ── colour palette ───────────────────────────────────────────────────────────
BLUE   = "#1a365d";  ACCENT = "#2b6cb0";  RED  = "#c53030"
GREEN  = "#276749";  ORANGE = "#c05621";  GRAY = "#718096"
PURPLE = "#553c9a";  TEAL   = "#234e52"

MODEL_COLOURS = [BLUE, ACCENT, TEAL, RED, ORANGE, PURPLE]

# ── model catalogue ──────────────────────────────────────────────────────────
MODELS = [
    {"id": "GARCH-N",   "label": "GARCH(1,1)\n+ Normal",          "vol": "GARCH",  "o": 0, "dist": "normal"},
    {"id": "GARCH-t",   "label": "GARCH(1,1)\n+ Student-t",        "vol": "GARCH",  "o": 0, "dist": "t"},
    {"id": "GARCH-St",  "label": "GARCH(1,1)\n+ Skewed-t",         "vol": "GARCH",  "o": 0, "dist": "skewt"},
    {"id": "GJR-t",     "label": "GJR-GARCH(1,1)\n+ Student-t",   "vol": "GARCH",  "o": 1, "dist": "t"},
    {"id": "GJR-St",    "label": "GJR-GARCH(1,1)\n+ Skewed-t",    "vol": "GARCH",  "o": 1, "dist": "skewt"},
    {"id": "EGARCH-t",  "label": "EGARCH(1,1)\n+ Student-t",       "vol": "EGARCH", "o": 0, "dist": "t"},
]

SEED = 42

# ── helpers ──────────────────────────────────────────────────────────────────
def _b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def kupiec_test(n_exceed: int, n_total: int, alpha: float) -> tuple[float, float]:
    """
    Kupiec proportion-of-failures (POF) test.
    H0: true exceedance rate == alpha (model is correctly calibrated).
    p > 0.05 → PASS (cannot reject correct coverage).
    """
    p  = alpha
    x  = n_exceed
    n  = n_total
    p_hat = x / n if x > 0 else 1e-10
    p_hat = min(p_hat, 1 - 1e-10)
    lr = -2.0 * (x * np.log(p / p_hat) + (n - x) * np.log((1 - p) / (1 - p_hat)))
    pval = float(stats.chi2.sf(lr, df=1))
    return float(lr), pval


def var_es_from_std_resid(
    std_resid: np.ndarray, cond_vol: np.ndarray, mu: float, level: float
) -> tuple[np.ndarray, float, float]:
    """
    Compute time-varying VaR and scalar ES from standardised residuals.
    Uses empirical quantile of std residuals (distribution-agnostic).
    Returns (var_series, mean_var, es).
    """
    q = float(np.percentile(std_resid, (1 - level) * 100))
    var_series = -(mu + cond_vol * q)
    r_pool = mu + cond_vol * std_resid
    es = float(-np.mean(r_pool[r_pool <= np.percentile(r_pool, (1 - level) * 100)]))
    return var_series, float(np.mean(var_series)), es


# ── Step 1: Fit all models ────────────────────────────────────────────────────
def fit_all(returns: pd.Series) -> list[dict]:
    results = []
    for i, m in enumerate(MODELS):
        print(f"    [{i+1}/{len(MODELS)}] {m['id']:12s} ...", end=" ", flush=True)
        try:
            am = arch_model(
                returns,
                mean="Constant",
                vol=m["vol"],
                p=1, o=m["o"], q=1,
                dist=m["dist"],
            )
            res = am.fit(disp="off", show_warning=False)
            std_r = res.std_resid.dropna().values
            cond_v = res.conditional_volatility.dropna().values
            mu = float(res.params.get("mu", res.params.get("Const", 0.0)))

            results.append({
                "meta":     m,
                "result":   res,
                "aic":      res.aic,
                "bic":      res.bic,
                "loglik":   res.loglikelihood,
                "nparams":  len(res.params),
                "std_resid": std_r,
                "cond_vol":  cond_v,
                "mu":        mu,
                "converged": True,
            })
            print(f"AIC={res.aic:.1f}  BIC={res.bic:.1f}  ok")
        except Exception as e:
            print(f"FAILED — {e}")
            results.append({"meta": m, "converged": False})

    return [r for r in results if r.get("converged")]


# ── Step 2: Residual diagnostics ─────────────────────────────────────────────
def residual_diagnostics(r: dict) -> dict:
    std  = r["std_resid"]
    lb20 = acorr_ljungbox(std,    lags=[20], return_df=True)["lb_pvalue"].iloc[0]
    lb20_sq = acorr_ljungbox(std**2, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
    jb_stat, jb_p = stats.jarque_bera(std)
    return {
        "lb_std_p":    float(lb20),        # > 0.05 = no serial corr = PASS
        "lb_sq_p":     float(lb20_sq),     # > 0.05 = no ARCH effect left = PASS
        "jb_p":        float(jb_p),        # > 0.05 = residuals normal = PASS
        "resid_kurt":  float(stats.kurtosis(std, fisher=True)),
        "resid_skew":  float(stats.skew(std)),
    }


# ── Step 3: VaR coverage (Kupiec) ────────────────────────────────────────────
def var_coverage(r: dict, real_returns: pd.Series) -> dict:
    ret   = real_returns.dropna().values[-len(r["std_resid"]):]
    cond  = r["cond_vol"]
    std   = r["std_resid"]
    mu    = r["mu"]
    n     = len(ret)

    var95_s, mean_var95, es95 = var_es_from_std_resid(std, cond, mu, 0.95)
    var99_s, mean_var99, es99 = var_es_from_std_resid(std, cond, mu, 0.99)

    exc95 = int(np.sum(ret < -var95_s))
    exc99 = int(np.sum(ret < -var99_s))

    lr95, p95 = kupiec_test(exc95, n, 0.05)
    lr99, p99 = kupiec_test(exc99, n, 0.01)

    return {
        "n":        n,
        "exc95":    exc95,   "rate95":  exc95 / n,
        "exc99":    exc99,   "rate99":  exc99 / n,
        "kup95_p":  p95,                          # > 0.05 = PASS
        "kup99_p":  p99,
        "mean_var95": mean_var95, "es95": es95,
        "mean_var99": mean_var99, "es99": es99,
    }


# ── Step 4: Simulate and compare to historical ────────────────────────────────
def simulate_compare(r: dict, real_returns: pd.Series,
                     n_paths: int = 500, horizon: int = 252) -> dict:
    res    = r["result"]
    real_r = real_returns.dropna().values

    # seed per-model from master seed
    master = np.random.default_rng(SEED)
    path_seeds = master.integers(0, 2**31, size=n_paths)

    paths = np.empty((n_paths, horizon))
    for i, s in enumerate(path_seeds):
        res.model.distribution._generator = np.random.default_rng(int(s))
        sim = res.model.simulate(res.params, nobs=horizon, burn=500)
        paths[i] = sim["data"].values

    synth_pool = paths.ravel()

    def var_es(x, lvl):
        q = np.percentile(x, (1 - lvl) * 100)
        return -q, -float(np.mean(x[x <= q]))

    r_var95, r_es95 = var_es(real_r, 0.95)
    r_var99, r_es99 = var_es(real_r, 0.99)
    s_var95, s_es95 = var_es(synth_pool, 0.95)
    s_var99, s_es99 = var_es(synth_pool, 0.99)

    def relerr(real, synth):
        return abs(synth / real - 1) if abs(real) > 1e-10 else abs(synth)

    return {
        "paths":            paths,
        "real_kurt":        float(stats.kurtosis(real_r, fisher=True)),
        "synth_kurt":       float(stats.kurtosis(synth_pool, fisher=True)),
        "kurt_err":         relerr(stats.kurtosis(real_r, fisher=True),
                                   stats.kurtosis(synth_pool, fisher=True)),
        "real_std":         float(real_r.std()),
        "synth_std":        float(synth_pool.std()),
        "std_err":          relerr(real_r.std(), synth_pool.std()),
        "real_skew":        float(stats.skew(real_r)),
        "synth_skew":       float(stats.skew(synth_pool)),
        "real_var95":  r_var95, "synth_var95": s_var95, "var95_err": relerr(r_var95, s_var95),
        "real_var99":  r_var99, "synth_var99": s_var99, "var99_err": relerr(r_var99, s_var99),
        "real_es95":   r_es95,  "synth_es95":  s_es95,  "es95_err":  relerr(r_es95, s_es95),
        "real_es99":   r_es99,  "synth_es99":  s_es99,  "es99_err":  relerr(r_es99, s_es99),
    }


# ── Step 5: Composite scoring ─────────────────────────────────────────────────
def score_models(fitted: list[dict]) -> list[dict]:
    """
    Score each model. Lower score = better.

    Points:
      AIC rank   0–5  (0=best)
      BIC rank   0–5  (0=best)
      +2 per LB std residual FAIL  (serial corr remains)
      +2 per LB sq residual FAIL   (ARCH effect remains)
      +2 per Kupiec FAIL at 95%
      +3 per Kupiec FAIL at 99%
      +2 per simulation metric (var95, var99, es95, es99, kurtosis) with >15% error
    """
    # AIC / BIC ranks
    aics = [f["aic"] for f in fitted]
    bics = [f["bic"] for f in fitted]
    aic_ranks = np.argsort(np.argsort(aics))
    bic_ranks = np.argsort(np.argsort(bics))

    scored = []
    for i, f in enumerate(fitted):
        d  = f["diag"]
        v  = f["var"]
        s  = f["sim"]
        sc = int(aic_ranks[i]) + int(bic_ranks[i])

        # residual adequacy
        if d["lb_std_p"] <= 0.05: sc += 2
        if d["lb_sq_p"]  <= 0.05: sc += 2

        # VaR coverage
        if v["kup95_p"] <= 0.05: sc += 2
        if v["kup99_p"] <= 0.05: sc += 3

        # simulation quality
        for key in ["var95_err", "var99_err", "es95_err", "es99_err", "kurt_err"]:
            if s[key] > 0.15: sc += 2

        scored.append({**f, "score": sc, "aic_rank": int(aic_ranks[i]), "bic_rank": int(bic_ranks[i])})

    scored.sort(key=lambda x: x["score"])
    return scored


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_aic_bic(scored: list[dict]) -> str:
    ids   = [f["meta"]["id"]  for f in scored]
    aics  = [f["aic"]         for f in scored]
    bics  = [f["bic"]         for f in scored]
    x     = np.arange(len(ids))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(11, 4))
    b1 = ax.bar(x - w/2, aics, w, label="AIC", color=BLUE,   alpha=0.85)
    b2 = ax.bar(x + w/2, bics, w, label="BIC", color=ACCENT, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=9)
    ax.set_ylabel("Information criterion (lower = better)")
    ax.set_title("AIC / BIC Comparison — All Models", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.25, axis="y")

    # annotate winner
    best_aic_idx = np.argmin(aics)
    ax.annotate("★ Best AIC", xy=(x[best_aic_idx] - w/2, aics[best_aic_idx]),
                xytext=(0, -18), textcoords="offset points", ha="center",
                fontsize=7, color=RED)
    fig.tight_layout()
    return _b64(fig)


def fig_residual_acf(scored: list[dict]) -> str:
    n = len(scored)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    fig.suptitle("Standardised Residual ACF — All Models", fontsize=11, fontweight="bold")
    lags = range(1, 26)
    ci   = 1.96 / np.sqrt(len(scored[0]["std_resid"]))

    for col, f in enumerate(scored):
        std = f["std_resid"]
        from statsmodels.tsa.stattools import acf as sm_acf
        acf_r  = sm_acf(std,    nlags=25, fft=True)
        acf_r2 = sm_acf(std**2, nlags=25, fft=True)

        for row, (acf_vals, title, color) in enumerate([
            (acf_r,  "Std Resid", BLUE),
            (acf_r2, "Sq Resid",  RED),
        ]):
            ax = axes[row, col]
            ax.bar(list(lags), acf_vals[1:], color=color, alpha=0.7, width=0.8)
            ax.axhline( ci, color="black", lw=0.8, linestyle="--")
            ax.axhline(-ci, color="black", lw=0.8, linestyle="--")
            ax.axhline(0,   color="black", lw=0.5)
            ax.set_title(f["meta"]["id"] + f"\n{title}", fontsize=7)
            ax.set_ylim(-0.12, 0.22)
            ax.tick_params(labelsize=6)

    fig.tight_layout()
    return _b64(fig)


def fig_var_coverage(scored: list[dict], real_returns: pd.Series) -> str:
    ids      = [f["meta"]["id"]  for f in scored]
    rates95  = [f["var"]["rate95"] * 100 for f in scored]
    rates99  = [f["var"]["rate99"] * 100 for f in scored]
    x = np.arange(len(ids)); w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("VaR Coverage — Actual vs Expected Exceedance Rate", fontsize=11, fontweight="bold")

    for ax, rates, expected, title in [
        (axes[0], rates95, 5.0, "VaR 95% — expected 5.0% exceedances"),
        (axes[1], rates99, 1.0, "VaR 99% — expected 1.0% exceedances"),
    ]:
        colors = [GREEN if abs(r - expected) < 1.5 else RED for r in rates]
        ax.bar(x, rates, color=colors, alpha=0.8)
        ax.axhline(expected, color="black", lw=2, linestyle="--",
                   label=f"Target {expected}%")
        ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=8)
        ax.set_ylabel("Exceedance rate (%)"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    return _b64(fig)


def fig_distribution_overlay(scored: list[dict], real_returns: pd.Series) -> str:
    r_real = real_returns.dropna().values
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Historical vs Synthetic Return Distributions — All Models",
                 fontsize=11, fontweight="bold")
    axes_flat = axes.flatten()

    from scipy.stats import gaussian_kde
    grid = np.linspace(-15, 15, 400)
    kde_real = gaussian_kde(r_real)(grid)

    for idx, f in enumerate(scored):
        ax = axes_flat[idx]
        synth = f["sim"]["paths"].ravel()
        kde_s = gaussian_kde(synth)(grid)

        ax.fill_between(grid, kde_real, alpha=0.3, color=BLUE,  label="Historical")
        ax.fill_between(grid, kde_s,    alpha=0.3, color=RED,   label="Synthetic")
        ax.plot(grid, kde_real, color=BLUE, lw=1.5)
        ax.plot(grid, kde_s,    color=RED,  lw=1.5, linestyle="--")
        ax.set_title(f["meta"]["id"], fontsize=9, fontweight="bold")
        ax.set_xlim(-15, 15); ax.set_xlabel("Return (%)", fontsize=7)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        if idx == 0: ax.set_ylabel("Density")

    fig.tight_layout()
    return _b64(fig)


def fig_cond_vol_comparison(scored: list[dict], real_returns: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle("Conditional Volatility — All Models Overlaid", fontsize=11, fontweight="bold")

    idx = real_returns.dropna().index
    for i, f in enumerate(scored):
        vol = f["cond_vol"]
        n   = min(len(vol), len(idx))
        ax.plot(idx[-n:], vol[-n:] * np.sqrt(252), lw=0.8,
                color=MODEL_COLOURS[i], alpha=0.75, label=f["meta"]["id"])

    ax.set_ylabel("Annualised conditional vol (%)")
    ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _b64(fig)


def fig_score_summary(scored: list[dict]) -> str:
    ids    = [f["meta"]["id"]  for f in scored]
    scores = [f["score"]       for f in scored]
    colors = [GREEN if i == 0 else (ORANGE if i == 1 else GRAY)
              for i in range(len(scored))]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(ids, scores, color=colors, alpha=0.85)
    ax.set_ylabel("Composite score (lower = better)")
    ax.set_title("Model Ranking — Composite Score", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    for bar, s in zip(bars, scores):
        ax.annotate(str(s), xy=(bar.get_x() + bar.get_width()/2, s),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.annotate("★ Winner", xy=(bars[0].get_x() + bars[0].get_width()/2, scores[0]),
                xytext=(0, 18), textcoords="offset points", ha="center",
                fontsize=8, color=GREEN)
    fig.tight_layout()
    return _b64(fig)


# ── HTML report ───────────────────────────────────────────────────────────────
_CSS = """
<style>
:root{--primary:#1a365d;--accent:#2b6cb0;--border:#e2e8f0;--text:#2d3748;--muted:#718096}
*{box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;color:var(--text);background:#f7fafc;
     max-width:1300px;margin:0 auto;padding:2rem 1.5rem;line-height:1.65}
h1{color:var(--primary);border-bottom:3px solid var(--accent);padding-bottom:.5rem;margin-top:0}
h2{color:var(--primary);margin-top:2.5rem;border-left:4px solid var(--accent);padding-left:.75rem}
h3{color:var(--accent);margin-top:1.4rem}
table{border-collapse:collapse;width:100%;margin:1rem 0;background:white;
      border-radius:6px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}
th{background:var(--primary);color:white;padding:.55rem 1rem;text-align:left;font-size:.83rem}
td{padding:.45rem .9rem;border-bottom:1px solid var(--border);font-size:.84rem}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:#f7fafc}
.pass{color:#276749;font-weight:700}
.fail{color:#9b2c2c;font-weight:700}
.winner-badge{display:inline-block;background:#f0fff4;border:2px solid #9ae6b4;
              border-radius:8px;padding:.75rem 1.5rem;font-size:1.1rem;
              font-weight:700;color:#276749;margin:.5rem 0}
figure{margin:1.5rem 0;background:white;padding:.75rem;
       border:1px solid var(--border);border-radius:6px}
figure img{max-width:100%;display:block;margin:0 auto}
figcaption{color:var(--muted);font-size:.8rem;margin-top:.4rem;text-align:center}
.insight{background:#ebf8ff;border-left:4px solid var(--accent);
         border-radius:0 6px 6px 0;padding:.9rem 1.2rem;margin:1rem 0;font-size:.9rem}
.warn{background:#fffbeb;border-left:4px solid #f6ad55;
      border-radius:0 6px 6px 0;padding:.9rem 1.2rem;margin:1rem 0;font-size:.9rem}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);
       color:var(--muted);font-size:.8rem}
</style>
"""

def p(v: bool) -> str:
    return '<span class="pass">PASS</span>' if v else '<span class="fail">FAIL</span>'


def build_html(scored: list[dict], real_returns: pd.Series, figs: dict) -> str:
    winner = scored[0]
    w_id   = winner["meta"]["id"]

    rows_fit = ""
    for f in scored:
        star = "★ " if f["meta"]["id"] == w_id else ""
        rows_fit += f"""<tr>
          <td><b>{star}{f['meta']['id']}</b></td>
          <td>{f['aic']:.1f}</td><td>{f['bic']:.1f}</td>
          <td>{f['loglik']:.1f}</td><td>{f['nparams']}</td>
          <td>{f['score']}</td>
        </tr>"""

    rows_diag = ""
    for f in scored:
        d = f["diag"]
        rows_diag += f"""<tr>
          <td><b>{f['meta']['id']}</b></td>
          <td>{d['lb_std_p']:.4f} &nbsp; {p(d['lb_std_p'] > 0.05)}</td>
          <td>{d['lb_sq_p']:.4f} &nbsp; {p(d['lb_sq_p'] > 0.05)}</td>
          <td>{d['resid_skew']:.3f}</td>
          <td>{d['resid_kurt']:.3f}</td>
          <td>{d['jb_p']:.4f} &nbsp; {p(d['jb_p'] > 0.05)}</td>
        </tr>"""

    rows_var = ""
    for f in scored:
        v = f["var"]
        rows_var += f"""<tr>
          <td><b>{f['meta']['id']}</b></td>
          <td>{v['exc95']} ({v['rate95']*100:.2f}%) &nbsp; {p(v['kup95_p'] > 0.05)}</td>
          <td>{v['kup95_p']:.4f}</td>
          <td>{v['exc99']} ({v['rate99']*100:.2f}%) &nbsp; {p(v['kup99_p'] > 0.05)}</td>
          <td>{v['kup99_p']:.4f}</td>
        </tr>"""

    rows_sim = ""
    for f in scored:
        s = f["sim"]
        def _re(v, t=0.15):
            cls = "pass" if v <= t else "fail"
            return f'<span class="{cls}">{v*100:.1f}%</span>'
        rows_sim += f"""<tr>
          <td><b>{f['meta']['id']}</b></td>
          <td>{s['real_std']:.3f} → {s['synth_std']:.3f} &nbsp; {_re(s['std_err'])}</td>
          <td>{s['real_kurt']:.2f} → {s['synth_kurt']:.2f} &nbsp; {_re(s['kurt_err'], 0.30)}</td>
          <td>{s['real_skew']:.3f} → {s['synth_skew']:.3f}</td>
          <td>{s['real_var95']:.3f} → {s['synth_var95']:.3f} &nbsp; {_re(s['var95_err'])}</td>
          <td>{s['real_var99']:.3f} → {s['synth_var99']:.3f} &nbsp; {_re(s['var99_err'])}</td>
          <td>{s['real_es99']:.3f} → {s['synth_es99']:.3f} &nbsp; {_re(s['es99_err'], 0.20)}</td>
        </tr>"""

    # build winner rationale from data
    wd = winner["diag"];  wv = winner["var"];  ws = winner["sim"]
    rationale = []
    rationale.append(f"Lowest composite score ({winner['score']} points) across all 4 scoring dimensions.")
    if wd["lb_sq_p"] > 0.05:
        rationale.append("Residual adequacy: no remaining ARCH effect in squared standardised residuals (LB p = {:.3f}).".format(wd["lb_sq_p"]))
    if wv["kup95_p"] > 0.05 and wv["kup99_p"] > 0.05:
        rationale.append(f"VaR coverage: both 95% and 99% Kupiec tests pass — correct exceedance rates.")
    if ws["var99_err"] < 0.15:
        rationale.append(f"Simulation quality: VaR99 error = {ws['var99_err']*100:.1f}% (within 15% tolerance).")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Model Selection — Brent Crude</title>
  {_CSS}
</head>
<body>
<h1>Brent Crude — Model Selection Report</h1>
<p style="color:var(--muted);font-size:.85rem;">
  Data: {real_returns.dropna().index.min().date()} → {real_returns.dropna().index.max().date()}
  &nbsp;|&nbsp; {len(real_returns.dropna()):,} observations
  &nbsp;|&nbsp; Generated: {generated}
</p>

<h2>0. Models Compared</h2>
<table>
  <thead><tr><th>ID</th><th>Volatility spec</th><th>Innovation dist.</th>
              <th>Captures leverage?</th><th>Captures skewness?</th></tr></thead>
  <tbody>
    <tr><td>GARCH-N</td>   <td>GARCH(1,1)</td>      <td>Normal</td>           <td class="fail">No</td><td class="fail">No</td></tr>
    <tr><td>GARCH-t</td>   <td>GARCH(1,1)</td>      <td>Student-t</td>        <td class="fail">No</td><td class="fail">No</td></tr>
    <tr><td>GARCH-St</td>  <td>GARCH(1,1)</td>      <td>Skewed Student-t</td> <td class="fail">No</td><td class="pass">Yes</td></tr>
    <tr><td>GJR-t</td>     <td>GJR-GARCH(1,1)</td>  <td>Student-t</td>        <td class="pass">Yes</td><td class="fail">No</td></tr>
    <tr><td>GJR-St</td>    <td>GJR-GARCH(1,1)</td>  <td>Skewed Student-t</td> <td class="pass">Yes</td><td class="pass">Yes</td></tr>
    <tr><td>EGARCH-t</td>  <td>EGARCH(1,1)</td>     <td>Student-t</td>        <td class="pass">Yes</td><td class="fail">No</td></tr>
  </tbody>
</table>
<div class="insight">
  <b>Why these 6?</b> The EDA showed three clear properties requiring modelling:
  (1) volatility clustering → GARCH family,
  (2) heavy tails → non-Gaussian innovations,
  (3) leverage asymmetry → GJR or EGARCH.
  The baseline (GARCH-N) serves as a null hypothesis — if it performs comparably,
  the added complexity is not warranted.
</div>

<h2>1. In-Sample Fit (AIC / BIC)</h2>
<figure>
  <img src="data:image/png;base64,{figs['aic_bic']}" alt="AIC BIC">
  <figcaption>Figure 1 — AIC and BIC for all 6 models. Lower = better fit adjusted for complexity.</figcaption>
</figure>
<table>
  <thead><tr><th>Model</th><th>AIC</th><th>BIC</th><th>Log-Likelihood</th>
              <th>Params</th><th>Composite Score</th></tr></thead>
  <tbody>{rows_fit}</tbody>
</table>

<h2>2. Residual Adequacy</h2>
<p style="font-size:.88rem;">
  After fitting, the standardised residuals (ẑ_t = ε_t / σ_t) should be approximately i.i.d.
  A PASS on the Ljung-Box tests means the model has absorbed all serial correlation
  (LB-std) and all volatility clustering (LB-sq). Remaining structure = the model is mis-specified.
</p>
<figure>
  <img src="data:image/png;base64,{figs['resid_acf']}" alt="Residual ACF">
  <figcaption>Figure 2 — ACF of standardised residuals (top) and squared standardised residuals (bottom)
  for each model. Bars inside the dashed lines (95% CI) indicate no remaining structure.</figcaption>
</figure>
<table>
  <thead><tr><th>Model</th><th>LB std resid (p)</th><th>LB sq resid (p)</th>
              <th>Resid skew</th><th>Resid kurtosis</th><th>Jarque-Bera (p)</th></tr></thead>
  <tbody>{rows_diag}</tbody>
</table>
<div class="insight">
  <b>Key:</b> LB sq resid PASS = GARCH captured all volatility clustering.
  LB sq resid FAIL = model missed some ARCH structure (common for simpler specs).
  Residual skewness near 0 and kurtosis near 0 = innovation distribution is well-specified.
</div>

<h2>3. VaR Coverage — Kupiec Test</h2>
<p style="font-size:.88rem;">
  The Kupiec POF test checks whether the model's VaR produces the correct exceedance rate.
  For VaR95: expect 5% of days below VaR (FAIL rate too high = under-estimates risk;
  too low = over-estimates risk). p &gt; 0.05 = cannot reject correct calibration = PASS.
</p>
<figure>
  <img src="data:image/png;base64,{figs['var_coverage']}" alt="VaR coverage">
  <figcaption>Figure 3 — Actual exceedance rates vs target (5% for VaR95, 1% for VaR99).
  Green bars are within ±1.5pp of target; red bars are outside.</figcaption>
</figure>
<table>
  <thead><tr><th>Model</th><th>VaR95 exceedances</th><th>Kupiec p (95%)</th>
              <th>VaR99 exceedances</th><th>Kupiec p (99%)</th></tr></thead>
  <tbody>{rows_var}</tbody>
</table>

<h2>4. Simulation Quality</h2>
<figure>
  <img src="data:image/png;base64,{figs['dist_overlay']}" alt="Distribution overlay">
  <figcaption>Figure 4 — Historical (blue) vs synthetic (red dashed) return distributions for each model.
  Good models produce nearly identical shapes in both centre and tails.</figcaption>
</figure>
<table>
  <thead><tr><th>Model</th><th>Std (real→synth)</th><th>Kurtosis (real→synth)</th>
              <th>Skewness (real→synth)</th>
              <th>VaR95 (real→synth)</th><th>VaR99 (real→synth)</th>
              <th>ES99 (real→synth)</th></tr></thead>
  <tbody>{rows_sim}</tbody>
</table>
<p style="font-size:.8rem;color:var(--muted);">
  Green = relative error within tolerance (std/VaR/ES: 15%; kurtosis: 30%; ES99: 20%).
  Red = outside tolerance.
</p>

<h2>5. Conditional Volatility</h2>
<figure>
  <img src="data:image/png;base64,{figs['cond_vol']}" alt="Conditional volatility">
  <figcaption>Figure 5 — Annualised conditional volatility from all 6 models overlaid.
  Models with leverage (GJR, EGARCH) show higher spikes after negative shocks.</figcaption>
</figure>

<h2>6. Composite Score & Winner</h2>
<figure>
  <img src="data:image/png;base64,{figs['score_bar']}" alt="Score">
  <figcaption>Figure 6 — Composite score. Lower = better. Score accumulates penalty points
  for AIC/BIC rank, residual failures, Kupiec failures, and simulation errors.</figcaption>
</figure>

<div class="winner-badge">★ Selected model: {w_id}</div>
<ul>
  {''.join(f'<li>{r}</li>' for r in rationale)}
</ul>

<div class="insight">
  <b>Decision rationale:</b> The scoring penalises missing structure (residual ARCH effects),
  poor VaR calibration, and simulation inaccuracy — not just in-sample fit.
  A model can have a good AIC but still fail if its residuals show remaining clustering
  or if its VaR coverage is systematically wrong.
  The winner balances all four dimensions.
</div>

<div class="warn">
  <b>Known limitations of the selected model ({w_id}):</b> Even the best model from this
  family (GARCH-based) cannot capture volatility regime shifts (COVID 2020, Ukraine 2022
  sustained-high-vol periods) or produce negatively-skewed innovations unless Skewed-t is
  selected. These are structural limitations of the model family, not implementation errors.
</div>

<footer>Brent Crude Model Selection · Generated {generated}</footer>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--n-paths", type=int, default=500)
    args = parser.parse_args()

    Path("reports").mkdir(exist_ok=True)

    print("\nBrent Crude — Model Selection")
    print(f"  start={args.start}  n_paths={args.n_paths}\n")

    # ── data
    print("  [1/6] Fetching data ...", end=" ", flush=True)
    from brent_stress.data import fetch_brent
    df      = fetch_brent(start=args.start)
    returns = df["log_return"].dropna()
    print(f"done  ({len(returns):,} days)")

    # ── fit
    print("  [2/6] Fitting models ...")
    fitted = fit_all(returns)
    print(f"        {len(fitted)}/{len(MODELS)} models converged")

    # ── diagnostics
    print("  [3/6] Residual diagnostics ...", end=" ", flush=True)
    for f in fitted:
        f["diag"] = residual_diagnostics(f)
    print("done")

    # ── VaR coverage
    print("  [4/6] VaR coverage (Kupiec) ...", end=" ", flush=True)
    for f in fitted:
        f["var"] = var_coverage(f, returns)
    print("done")

    # ── simulation
    print(f"  [5/6] Simulating {args.n_paths} paths per model ...")
    for i, f in enumerate(fitted):
        print(f"        [{i+1}/{len(fitted)}] {f['meta']['id']} ...", end=" ", flush=True)
        f["sim"] = simulate_compare(f, returns, n_paths=args.n_paths)
        print("done")

    # ── score
    print("  [6/6] Scoring and ranking ...", end=" ", flush=True)
    scored = score_models(fitted)
    print("done")

    print("\n  === Model Ranking ===")
    for rank, f in enumerate(scored, 1):
        print(f"  #{rank}  {f['meta']['id']:12s}  score={f['score']:2d}  "
              f"AIC={f['aic']:.0f}  BIC={f['bic']:.0f}")
    print(f"\n  ★ Winner: {scored[0]['meta']['id']}")

    # ── figures
    print("\n  Generating figures ...", end=" ", flush=True)
    figs = {
        "aic_bic":     fig_aic_bic(scored),
        "resid_acf":   fig_residual_acf(scored),
        "var_coverage":fig_var_coverage(scored, returns),
        "dist_overlay":fig_distribution_overlay(scored, returns),
        "cond_vol":    fig_cond_vol_comparison(scored, returns),
        "score_bar":   fig_score_summary(scored),
    }
    print("done")

    # ── report
    print("  Writing reports/model_selection.html ...", end=" ", flush=True)
    html = build_html(scored, returns, figs)
    Path("reports/model_selection.html").write_text(html, encoding="utf-8")
    print("done")

    print("\n  → Open reports/model_selection.html in your browser\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
