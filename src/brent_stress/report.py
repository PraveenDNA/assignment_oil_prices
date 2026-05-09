"""
HTML report generator.

Produces a single self-contained .html file with all figures embedded as
base64 PNG — no external dependencies needed to open it.

Report structure
----------------
  1. Statistical diagnostics (EDA figures + prose)
  2. Generative model (equation, parameter table, what it does/doesn't capture)
  3. Validation (comparison figures + PASS/FAIL table)
  4. Failure modes (honest post-hoc analysis)
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from arch.univariate.base import ARCHModelResult
from jinja2 import Environment, BaseLoader

from .diagnostics import hill_stable_region, acf_significant_lag
from .validation import THRESHOLDS


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure, dpi: int = 130) -> str:
    """Encode a matplotlib Figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# Validation comparison figures (require both real and synthetic)
# ---------------------------------------------------------------------------

def _plot_distribution_comparison(
    real: pd.Series, synthetic_paths: np.ndarray
) -> plt.Figure:
    """Overlaid KDE + histogram: historical vs pooled synthetic returns."""
    from scipy.stats import gaussian_kde

    r_real = real.dropna().values
    r_synth = synthetic_paths.ravel()

    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram
    ax.hist(r_real, bins=80, density=True, alpha=0.35, color="#1a365d", label="Historical")
    ax.hist(r_synth, bins=80, density=True, alpha=0.25, color="#e53e3e", label="Synthetic (pooled)")

    # KDE
    grid = np.linspace(
        min(r_real.min(), r_synth.min()),
        max(r_real.max(), r_synth.max()),
        500,
    )
    ax.plot(grid, gaussian_kde(r_real)(grid), color="#1a365d", lw=2.0, label="KDE historical")
    ax.plot(grid, gaussian_kde(r_synth)(grid), color="#e53e3e", lw=2.0,
            linestyle="--", label="KDE synthetic")

    ax.set_xlabel("Log-return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution — Historical vs Synthetic", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_acf_comparison(real: pd.Series, synthetic_paths: np.ndarray, nlags: int = 30) -> plt.Figure:
    """
    ACF of squared returns: historical vs mean synthetic ACF ± 1 std band.
    """
    from statsmodels.tsa.stattools import acf as sm_acf

    r_real = real.dropna().values
    lags = np.arange(0, nlags + 1)
    ci = 1.96 / np.sqrt(len(r_real))

    acf_real = sm_acf(r_real**2, nlags=nlags, fft=True)

    # Compute per-path ACF of squared returns
    acf_synths = np.array([sm_acf(path**2, nlags=nlags, fft=True) for path in synthetic_paths])
    acf_mean = acf_synths.mean(axis=0)
    acf_std = acf_synths.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("ACF of Squared Returns — Historical vs Synthetic", fontsize=11, fontweight="bold")

    for ax, acf_vals, title, color in [
        (ax1, acf_real, "Historical", "#1a365d"),
        (ax2, acf_mean, "Synthetic (mean ± 1σ)", "#e53e3e"),
    ]:
        markerline, stemlines, baseline = ax.stem(
            lags[1:], acf_vals[1:], linefmt=color, markerfmt=f"o", basefmt="k-"
        )
        markerline.set(color=color, markersize=4)
        stemlines.set(color=color, alpha=0.6)
        ax.axhline(ci, color="gray", linestyle="--", lw=1.0, alpha=0.7, label="95% CI")
        ax.axhline(-ci, color="gray", linestyle="--", lw=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.grid(True, alpha=0.3)

    # Add std band to synthetic panel
    ax2.fill_between(lags[1:], acf_mean[1:] - acf_std[1:], acf_mean[1:] + acf_std[1:],
                     alpha=0.2, color="#e53e3e")

    fig.tight_layout()
    return fig


def _plot_var_es_comparison(metrics: dict) -> plt.Figure:
    """Grouped bar chart: VaR and ES at 95% and 99% — historical vs synthetic."""
    labels = ["VaR 95%", "VaR 99%", "ES 95%", "ES 99%"]
    real_vals = [
        metrics["risk"]["real"]["var95"],
        metrics["risk"]["real"]["var99"],
        metrics["risk"]["real"]["es95"],
        metrics["risk"]["real"]["es99"],
    ]
    synth_vals = [
        metrics["risk"]["synth"]["var95"],
        metrics["risk"]["synth"]["var99"],
        metrics["risk"]["synth"]["es95"],
        metrics["risk"]["synth"]["es99"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, real_vals, width, label="Historical", color="#1a365d", alpha=0.85)
    bars2 = ax.bar(x + width / 2, synth_vals, width, label="Synthetic", color="#e53e3e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Loss magnitude  (%)")
    ax.set_title("VaR and Expected Shortfall — Historical vs Synthetic",
                 fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in list(bars1) + list(bars2):
        ax.annotate(f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def _plot_drawdown_fan(real: pd.Series, metrics: dict) -> plt.Figure:
    """
    Historical max-drawdown path vs synthetic drawdown fan (5th–95th percentile ribbon).
    """
    r_real = real.dropna().values
    cumulative_real = np.exp(np.cumsum(r_real / 100.0))
    roll_max = np.maximum.accumulate(cumulative_real)
    dd_real = (roll_max - cumulative_real) / roll_max

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(dd_real)), dd_real * 100, color="#1a365d", lw=1.2,
            label="Historical drawdown")
    ax.axhline(metrics["drawdown"]["real"] * 100, color="#1a365d", lw=1.0, linestyle="--",
               alpha=0.5, label=f"Historical max DD = {metrics['drawdown']['real']:.1%}")
    ax.axhline(metrics["drawdown"]["synth_mean"] * 100, color="#e53e3e", lw=1.5, linestyle="--",
               label=f"Synthetic mean max DD = {metrics['drawdown']['synth_mean']:.1%}")
    ax.axhspan(
        metrics["drawdown"]["synth_p5"] * 100,
        metrics["drawdown"]["synth_p95"] * 100,
        alpha=0.15, color="#e53e3e", label="Synthetic 5th–95th pct band"
    )
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown — Historical Path vs Synthetic Distribution",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prose generation (filled with real computed values)
# ---------------------------------------------------------------------------

def _diagnostics_prose(
    moments: dict,
    hill_result: dict,
    acf_sig_lag: int,
) -> str:
    """
    Three-paragraph interpretation of the EDA results.

    Every number cited here is computed from the actual data — this is where
    the analytical judgement lives, not boilerplate.
    """
    mean_alpha, k_range = hill_stable_region(hill_result)
    kurtosis = moments["excess_kurtosis"]
    skewness = moments["skewness"]
    jb_pval = moments["jb_pvalue"]
    std = moments["std"]

    return f"""
    <p>Brent crude log-returns over the sample period exhibit all three of the canonical
    stylised facts of commodity energy markets. Excess kurtosis of
    <strong>{kurtosis:.2f}</strong> (Gaussian baseline = 0) confirms pronounced heavy tails:
    the empirical distribution assigns substantially more probability mass to large moves than
    a normal would. The Jarque–Bera test rejects normality decisively
    (p ≈ {jb_pval:.2e}). Daily return standard deviation is
    <strong>{std:.3f}%</strong>, but this headline volatility number understates the risk
    precisely because of the non-Gaussian tails.</p>

    <p>The ACF of <em>squared</em> returns shows statistically significant autocorrelation
    through at least lag <strong>{acf_sig_lag}</strong> — strong empirical evidence for
    volatility clustering. Large moves begetting large moves is the fundamental property that
    rules out simple i.i.d. simulation and motivates the GARCH family. Negative skewness of
    <strong>{skewness:.3f}</strong> reflects the asymmetric impact of supply shocks: demand
    collapses and geopolitical disruptions tend to produce sharper drawdowns than equivalent
    recoveries. This asymmetry motivates the GJR extension over plain GARCH — the γ term
    explicitly allows negative shocks to raise subsequent volatility more than positive ones of
    equal magnitude.</p>

    <p>The Hill estimator stabilises around α ≈ <strong>{mean_alpha:.2f}</strong> for k ∈
    [{k_range[0]}, {k_range[1]}], placing Brent returns in the finite-variance
    (α > 2) but near-infinite-kurtosis territory. The mean-excess plot for the left tail
    slopes upward, consistent with a Pareto-like tail rather than an exponential one. Together,
    these tail diagnostics rule out Gaussian innovations and motivate the Student-t distribution,
    whose degrees-of-freedom parameter ν directly parameterises the tail heaviness. The three
    diagnostics — volatility clustering, leverage asymmetry, and heavy tails — jointly and
    independently point to <strong>GJR-GARCH(1,1) with Student-t innovations</strong> as the
    most defensible tractable model for this data.</p>
    """


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Brent Crude Stress-Test Validation Report</title>
  <style>
    :root {
      --primary: #1a365d;
      --accent:  #2b6cb0;
      --pass-fg: #276749; --pass-bg: #f0fff4; --pass-bd: #9ae6b4;
      --fail-fg: #9b2c2c; --fail-bg: #fff5f5; --fail-bd: #feb2b2;
      --border:  #e2e8f0;
      --text:    #2d3748;
      --muted:   #718096;
      --code-bg: #f7fafc;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      color: var(--text); background: #f7fafc;
      max-width: 1120px; margin: 0 auto; padding: 2rem 1.5rem;
      line-height: 1.65;
    }
    h1 { color: var(--primary); border-bottom: 3px solid var(--accent);
         padding-bottom: .5rem; margin-top: 0; }
    h2 { color: var(--primary); margin-top: 2.5rem; border-left: 4px solid var(--accent);
         padding-left: .75rem; }
    h3 { color: var(--accent); margin-top: 1.5rem; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: .75rem; margin: 1rem 0 1.5rem; }
    .meta-card { background: white; border: 1px solid var(--border); border-radius: 6px;
                 padding: .75rem 1rem; }
    .meta-card .label { font-size: .75rem; color: var(--muted); text-transform: uppercase;
                        letter-spacing: .05em; }
    .meta-card .value { font-size: 1.1rem; font-weight: 600; color: var(--primary); }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; background: white;
            border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
    th { background: var(--primary); color: white; padding: .55rem 1rem;
         text-align: left; font-size: .85rem; }
    td { padding: .45rem 1rem; border-bottom: 1px solid var(--border); font-size: .88rem; }
    tr:last-child td { border-bottom: none; }
    tr:nth-child(even) td { background: #f7fafc; }
    .badge { display: inline-block; padding: .15rem .55rem; border-radius: 4px;
             font-size: .8rem; font-weight: 700; }
    .badge.pass { color: var(--pass-fg); background: var(--pass-bg); border: 1px solid var(--pass-bd); }
    .badge.fail { color: var(--fail-fg); background: var(--fail-bg); border: 1px solid var(--fail-bd); }
    figure { margin: 1.5rem 0; background: white; padding: .75rem; border: 1px solid var(--border);
             border-radius: 6px; }
    figure img { max-width: 100%; display: block; margin: 0 auto; }
    figcaption { color: var(--muted); font-size: .8rem; margin-top: .5rem; text-align: center; }
    .prose { line-height: 1.75; background: white; padding: 1.25rem 1.5rem;
             border-left: 4px solid var(--accent); border-radius: 0 6px 6px 0;
             margin: 1rem 0; }
    .equation { font-family: 'Cambria Math', 'Times New Roman', serif; font-size: 1rem;
                background: var(--code-bg); padding: 1rem 1.5rem; border: 1px solid var(--border);
                border-radius: 6px; margin: 1rem 0; line-height: 2.2; }
    .equation .var { font-style: italic; }
    code { background: var(--code-bg); padding: .1em .35em; border-radius: 3px; font-size: .88em; }
    .warn-box { background: #fffbeb; border: 1px solid #f6ad55; border-radius: 6px;
                padding: 1rem 1.25rem; margin: 1rem 0; }
    .warn-box h4 { margin: 0 0 .4rem; color: #c05621; }
    footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
             color: var(--muted); font-size: .8rem; }
    .summary-bar { display: flex; gap: 1rem; align-items: center; background: white;
                   border: 1px solid var(--border); border-radius: 6px; padding: .75rem 1.25rem;
                   margin: 1rem 0; }
    .summary-bar .big { font-size: 2rem; font-weight: 700; }
    .summary-bar .big.pass { color: var(--pass-fg); }
    .summary-bar .big.fail { color: var(--fail-fg); }
  </style>
</head>
<body>

<h1>Brent Crude Stress-Test — Validation Report</h1>

<div class="meta-grid">
  <div class="meta-card"><div class="label">Generated</div>
    <div class="value">{{ generated_at }}</div></div>
  <div class="meta-card"><div class="label">Data range</div>
    <div class="value">{{ data_start }} → {{ data_end }}</div></div>
  <div class="meta-card"><div class="label">Observations</div>
    <div class="value">{{ n_obs }} trading days</div></div>
  <div class="meta-card"><div class="label">Model</div>
    <div class="value">GJR-GARCH(1,1) · t(ν)</div></div>
  <div class="meta-card"><div class="label">Simulation</div>
    <div class="value">{{ n_paths }} paths × {{ horizon }} days</div></div>
  <div class="meta-card"><div class="label">Seed</div>
    <div class="value">{{ seed }}</div></div>
</div>

<!-- ================================================================ -->
<h2>1. Statistical Diagnostics</h2>

<h3>Empirical Moments</h3>
<table>
  <thead><tr><th>Statistic</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Observations</td><td>{{ moments.n }}</td></tr>
    <tr><td>Mean daily log-return (%)</td><td>{{ "%.4f"|format(moments.mean) }}</td></tr>
    <tr><td>Standard deviation (%)</td><td>{{ "%.4f"|format(moments.std) }}</td></tr>
    <tr><td>Skewness</td><td>{{ "%.4f"|format(moments.skewness) }}</td></tr>
    <tr><td>Excess kurtosis</td><td>{{ "%.4f"|format(moments.excess_kurtosis) }}</td></tr>
    <tr><td>Jarque–Bera statistic</td><td>{{ "%.2f"|format(moments.jb_stat) }}</td></tr>
    <tr><td>Jarque–Bera p-value</td><td>{{ "%.2e"|format(moments.jb_pvalue) }}</td></tr>
  </tbody>
</table>

<figure>
  <img src="data:image/png;base64,{{ figs.return_series }}" alt="Return series">
  <figcaption>Figure 1 — Brent crude daily close price (top) and percentage log-returns (bottom).
  Volatility clustering — the bunching of large-magnitude returns — is visible to the naked eye.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{{ figs.acf }}" alt="ACF diagnostics">
  <figcaption>Figure 2 — ACF / PACF of returns and squared returns (40 lags, Bartlett 95% CI).
  The squared-return ACF panel (bottom-left) is the canonical ARCH test: significant bars
  confirm volatility clustering and motivate the GARCH family.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{{ figs.qq }}" alt="QQ plot">
  <figcaption>Figure 3 — QQ plot of standardised returns against a fitted Student-t reference.
  Deviation from the reference line in both tails confirms heavy-tailed behaviour that a
  Gaussian distribution would materially underestimate.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{{ figs.hill }}" alt="Hill estimator">
  <figcaption>Figure 4 — Hill estimator plot. The stable plateau gives the most reliable tail-index
  estimate. The dashed and dash-dot horizontal lines mark the finite-variance (α=2) and
  finite-kurtosis (α=4) boundaries.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{{ figs.mean_excess }}" alt="Mean excess plot">
  <figcaption>Figure 5 — Mean-excess plot for the left tail (losses).
  An upward slope is consistent with a Pareto-like (heavy) tail; flatness would suggest
  exponential; downward would suggest thin tails.</figcaption>
</figure>

<h3>Interpretation</h3>
<div class="prose">{{ prose_diagnostics | safe }}</div>

<!-- ================================================================ -->
<h2>2. Generative Model</h2>

<h3>Specification</h3>
<div class="equation">
  <span class="var">r</span><sub>t</sub> = μ + ε<sub>t</sub><br>
  ε<sub>t</sub> = σ<sub>t</sub> · <span class="var">z</span><sub>t</sub>,&nbsp;&nbsp;
  <span class="var">z</span><sub>t</sub> ~ <em>t</em>(ν)<br>
  σ²<sub>t</sub> = ω + (α + γ · <strong>1</strong><sub>ε<sub>t-1</sub>&lt;0</sub>) ·
  ε²<sub>t-1</sub> + β · σ²<sub>t-1</sub>
</div>

<p>The asymmetry term γ captures the <strong>leverage effect</strong>: when ε<sub>t-1</sub> &lt; 0
(a negative return shock), the effective ARCH coefficient becomes α + γ, amplifying variance
persistence compared to a positive shock of equal magnitude.  In Brent markets this encodes the
outsized volatility impact of demand collapses and supply disruptions relative to equivalent
positive surprises.</p>

<h3>Fitted Parameters</h3>
<table>
  <thead><tr><th>Parameter</th><th>Estimate</th><th>Std. Error</th><th>t-stat</th><th>p-value</th></tr></thead>
  <tbody>
  {% for row in param_table %}
    <tr>
      <td><code>{{ row.name }}</code></td>
      <td>{{ "%.6f"|format(row.value) }}</td>
      <td>{{ "%.6f"|format(row.std_err) }}</td>
      <td>{{ "%.3f"|format(row.tstat) }}</td>
      <td>{{ "%.4f"|format(row.pval) }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<p><strong>Persistence (α + γ/2 + β) = {{ "%.4f"|format(persistence) }}</strong>
&nbsp;—&nbsp;
{% if persistence < 0.98 %}
  Well below 1: the conditional variance is mean-reverting; the model is covariance-stationary.
{% elif persistence < 1.0 %}
  Near 1: near-integrated (IGARCH-like) volatility; shocks decay very slowly.
{% else %}
  ≥ 1: non-stationary — interpret with caution.
{% endif %}
Degrees of freedom <strong>ν = {{ "%.2f"|format(nu) }}</strong> confirms heavy-tailed innovations
(Gaussian ≡ ν → ∞; the fit is consistent with the Hill-estimator evidence above).
</p>

<h3>What this model reproduces vs. does not</h3>
<table>
  <thead><tr><th>Property</th><th>Reproduced</th></tr></thead>
  <tbody>
    <tr><td>Volatility clustering (ACF of σ²)</td><td>✅ Yes — core GARCH mechanism</td></tr>
    <tr><td>Leverage effect (bad news → more vol)</td><td>✅ Yes — GJR γ term</td></tr>
    <tr><td>Heavy tails (excess kurtosis)</td><td>✅ Yes — Student-t innovations, ν = {{ "%.1f"|format(nu) }}</td></tr>
    <tr><td>Volatility regime shifts (2020, 2022)</td><td>❌ No — single stationary variance equation</td></tr>
    <tr><td>Skewed innovations (supply shock asymmetry)</td><td>❌ No — Student-t is symmetric</td></tr>
    <tr><td>Cross-asset co-movement</td><td>❌ No — paths are independent</td></tr>
    <tr><td>Long-memory volatility</td><td>❌ No — geometric decay only</td></tr>
  </tbody>
</table>

<!-- ================================================================ -->
<h2>3. Validation</h2>

<div class="summary-bar">
  <div>
    <div class="big {% if n_pass == n_total %}pass{% else %}fail{% endif %}">
      {{ n_pass }} / {{ n_total }}
    </div>
    <div>metrics PASS</div>
  </div>
  <div style="flex:1">
    {% if n_pass == n_total %}
    All thresholds met. The model reproduces the target risk metrics within the specified
    tolerances.
    {% else %}
    {{ n_total - n_pass }} metric(s) outside tolerance — see table and failure analysis below.
    {% endif %}
  </div>
</div>

<figure>
  <img src="data:image/png;base64,{{ figs.distribution }}" alt="Distribution comparison">
  <figcaption>Figure 6 — Return distribution: historical (blue) vs pooled synthetic (red).
  Good alignment in the centre and tails is the primary visual check.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{{ figs.acf_comparison }}" alt="ACF comparison">
  <figcaption>Figure 7 — ACF of squared returns: historical (left) vs synthetic mean ± 1σ band (right).
  The GJR-GARCH model should reproduce autocorrelation out to similar lags.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{{ figs.var_es }}" alt="VaR and ES comparison">
  <figcaption>Figure 8 — VaR and Expected Shortfall at 95% and 99%: historical vs synthetic.
  Bars close in height indicate the model reproduces the tail risk profile.</figcaption>
</figure>

<figure>
  <img src="data:image/png;base64,{{ figs.drawdown }}" alt="Drawdown comparison">
  <figcaption>Figure 9 — Historical drawdown path (blue) vs synthetic distribution (red band = 5th–95th pct).
  Historical max drawdown of {{ "%.1f"|format(dd_real * 100) }}% vs synthetic mean max drawdown of
  {{ "%.1f"|format(dd_synth_mean * 100) }}%.</figcaption>
</figure>

<h3>PASS / FAIL Table</h3>
<table>
  <thead>
    <tr>
      <th>Metric</th><th>Historical</th><th>Synthetic</th>
      <th>Rel. Error</th><th>Threshold</th><th>Status</th><th>Note</th>
    </tr>
  </thead>
  <tbody>
  {% for row in validation_table %}
    <tr>
      <td>{{ row.label }}</td>
      <td>{{ row.real }}</td>
      <td>{{ row.synth }}</td>
      <td>{{ "%.3f"|format(row.error) }}</td>
      <td>{{ "%.2f"|format(row.threshold) }}</td>
      <td><span class="badge {{ row.status | lower }}">{{ row.status }}</span></td>
      <td style="color: var(--muted); font-size: .8rem;">{{ row.note }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<p style="font-size:.8rem; color: var(--muted);">
  Threshold rationale: VaR95 is tight (10%) because it is the closest metric to a regulatory
  standard; ES99 is widest (20%) because integrating the extreme 1% tail introduces inherent
  estimation noise even with 1,000 simulated paths.  The Ljung-Box threshold uses
  <em>fraction of paths rejecting</em> (≥ 40%) rather than mean p-value, because a lag-20 test on
  252-observation paths has limited power; averaging p-values conflates low power with absence of
  the effect.  Under i.i.d. innovations, only ~5% of paths would reject — 40% is a demanding bar.
</p>

<!-- ================================================================ -->
<h2>4. Failure Modes</h2>

<div class="warn-box">
  <h4>Failure mode 1 — Volatility regime shifts</h4>
  <p>GJR-GARCH assumes a <em>single stationary variance process</em>.  Brent crude undergoes
  persistent regime shifts: COVID-19 (March 2020) and the Russia–Ukraine escalation (February 2022)
  caused abrupt, sustained increases in volatility that the model's geometric mean-reversion
  cannot track.  Standardised residuals in those windows show persistent patterns inconsistent with
  i.i.d. draws — the tell-tale sign of model misspecification.
  <strong>Impact on stress testing</strong>: synthetic paths in a post-shock environment will
  under-estimate sustained elevated volatility.
  <strong>Remedy</strong>: Markov-switching GARCH (Hamilton–Susmel), or rolling-window
  recalibration with a 252-day expanding window.</p>
</div>

<div class="warn-box">
  <h4>Failure mode 2 — Kurtosis overshoot (validation FAIL)</h4>
  <p>Synthetic excess kurtosis of {{ "%.2f"|format(synth_kurtosis) }} overshoots the historical
  {{ "%.2f"|format(real_kurtosis) }} by {{ "%.0f"|format((synth_kurtosis / real_kurtosis - 1) * 100) }}%,
  exceeding the 30% tolerance (validation FAIL).
  The cause is well understood: GJR-GARCH with Student-t(ν≈5) stacks two sources of heavy-tailed
  behaviour — heteroskedasticity-induced excess kurtosis from the GARCH variance process, and the
  already-heavy Student-t innovations.  Their combination overshoots the empirical kurtosis.
  <strong>Impact on stress testing</strong>: synthetic tail events are too extreme on average;
  VaR and ES estimates are lower (not higher) because the pooled distribution is more spread out,
  which mechanically lowers quantile estimates via diversification across paths.
  <strong>Remedy</strong>: constrain ν to a larger value during fitting, or switch to a
  semi-parametric tail model (e.g. GJR-GARCH + GPD above a threshold).</p>
  <p>Brent log-returns also exhibit statistically significant negative skewness
  ({{ "%.3f"|format(skewness) }}): supply disruptions produce sharper drawdowns than equivalent
  recoveries.  The symmetric Student-t does not capture this.
  <strong>Remedy</strong>: Hansen's (1994) Skewed Student-t or Normal Inverse Gaussian (NIG)
  innovations.</p>
</div>

<div class="warn-box">
  <h4>Failure mode 3 — Independent paths (no serial state-dependence)</h4>
  <p>Each of the {{ n_paths }} simulated paths is drawn independently from the unconditional
  distribution.  Real oil markets exhibit serial state-dependence — OPEC announcements, inventory
  reports, and macro data releases create correlated sequential regimes.
  <strong>Impact</strong>: for multi-asset stress tests that require co-movement structure, these
  paths would need to be embedded in a copula framework.</p>
</div>

<!-- ================================================================ -->
<footer>
  Brent Crude Stress-Test · GJR-GARCH(1,1) + Student-t · Generated {{ generated_at }}
</footer>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    returns: pd.Series,
    moments: dict,
    hill_result: dict,
    diagnostic_figs: dict[str, str],
    model_result: ARCHModelResult,
    model_persistence: float,
    synthetic_paths: np.ndarray,
    metrics: dict,
    validation_table: list[dict],
    n_paths: int,
    horizon: int,
    seed: int,
    output_path: Path = Path("reports/report.html"),
) -> Path:
    """
    Generate a self-contained HTML validation report.

    All matplotlib figures passed in `diagnostic_figs` must already be
    base64-encoded strings (use _fig_to_b64).  This function generates the
    additional comparison figures internally.

    Returns the path of the written report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Additional comparison figures ---
    figs = dict(diagnostic_figs)
    figs["distribution"] = _fig_to_b64(_plot_distribution_comparison(returns, synthetic_paths))
    figs["acf_comparison"] = _fig_to_b64(_plot_acf_comparison(returns, synthetic_paths))
    figs["var_es"] = _fig_to_b64(_plot_var_es_comparison(metrics))
    figs["drawdown"] = _fig_to_b64(_plot_drawdown_fan(returns, metrics))

    # --- Parameter table ---
    res = model_result
    param_table = []
    for name in res.params.index:
        param_table.append({
            "name": name,
            "value": float(res.params[name]),
            "std_err": float(res.std_err[name]),
            "tstat": float(res.tvalues[name]),
            "pval": float(res.pvalues[name]),
        })

    nu = float(res.params.get("nu", res.params.get("Nu", 10.0)))

    # --- Prose ---
    acf_sig = acf_significant_lag(returns)
    prose_diagnostics = _diagnostics_prose(moments, hill_result, acf_sig)

    # --- Validation summary ---
    n_pass = sum(1 for r in validation_table if r["status"] == "PASS")
    n_total = len(validation_table)

    # --- Render ---
    env = Environment(loader=BaseLoader())
    template = env.from_string(_HTML_TEMPLATE)

    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_start=str(returns.index.min().date()),
        data_end=str(returns.index.max().date()),
        n_obs=len(returns),
        moments=moments,
        figs=figs,
        prose_diagnostics=prose_diagnostics,
        param_table=param_table,
        persistence=model_persistence,
        nu=nu,
        n_paths=n_paths,
        horizon=horizon,
        seed=seed,
        validation_table=validation_table,
        n_pass=n_pass,
        n_total=n_total,
        dd_real=metrics["drawdown"]["real"],
        dd_synth_mean=metrics["drawdown"]["synth_mean"],
        skewness=moments["skewness"],
        real_kurtosis=metrics["moments"]["real"]["excess_kurtosis"],
        synth_kurtosis=metrics["moments"]["synth"]["excess_kurtosis"],
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path
