"""
Validation: compare synthetic return paths against historical Brent data.

Threshold calibration rationale
---------------------------------
Thresholds are set to be demanding enough to be meaningful for stress testing
but realistic given the inherent estimation noise in each metric.

  kurtosis_rel_error ≤ 0.30
    Excess kurtosis governs tail mass and therefore extreme-scenario
    frequency.  A 30% relative error still gives broadly correct tail
    behaviour; beyond that the model is systematically wrong about how
    extreme moves are.

  var95_rel_error ≤ 0.10
    The 95% VaR (Value-at-Risk) is a core regulatory metric under Basel III /
    FRTB.  A 10% tolerance means a ±$0.50 error on a $5 daily loss estimate
    — acceptable for internal stress testing but tight enough to be
    meaningful.

  var99_rel_error ≤ 0.15
    The 99% VaR threshold is more extreme; fewer historical observations lie
    in the relevant tail so estimation variance is unavoidably higher.
    15% is proportionally tighter than ES99 but looser than VaR95.

  es95_rel_error ≤ 0.15
    Expected Shortfall (CVaR) integrates all losses beyond VaR.  It is more
    sensitive to exact tail shape than VaR and inherently noisier to estimate
    from a finite sample.

  es99_rel_error ≤ 0.20
    The 99% ES integrates the most extreme 1% of the distribution.  With
    ~2,500 days of history that is only ~25 observations.  Synthetic
    averaging over 1,000 paths is more stable, but 20% tolerance honestly
    reflects how hard this metric is to nail precisely.

  lb_reject_frac ≥ 0.40
    The fraction of simulated paths (252 days each) for which the Ljung-Box
    test on squared returns rejects the null of no autocorrelation (at 5%)
    must exceed 40%.  Under the null of i.i.d. returns, only ~5% of paths
    would reject — so 40% is a demanding minimum.  A per-path test (rather
    than a pooled test) is used because it directly measures whether individual
    252-day windows exhibit the GARCH property; the fraction-of-rejection
    aggregation accounts for the limited power of a lag-20 test on 252
    observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


# ---------------------------------------------------------------------------
# Threshold definitions — edit with care; each threshold has a comment above
# explaining the calibration logic
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "kurtosis_rel_error": 0.30,
    "var95_rel_error": 0.10,
    "var99_rel_error": 0.15,
    "es95_rel_error": 0.15,
    "es99_rel_error": 0.20,
    # For lb_reject_frac: PASS means fraction of rejecting paths >= threshold
    "lb_reject_frac_min": 0.40,
}

THRESHOLD_LABELS: dict[str, str] = {
    "kurtosis_rel_error": "Excess kurtosis relative error",
    "var95_rel_error": "VaR 95% relative error",
    "var99_rel_error": "VaR 99% relative error",
    "es95_rel_error": "ES 95% relative error",
    "es99_rel_error": "ES 99% relative error",
    "lb_reject_frac_min": "LB reject fraction (sq. returns, p<0.05)",
}


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_var_es(returns: np.ndarray, level: float) -> tuple[float, float]:
    """
    Historical (empirical) VaR and Expected Shortfall at confidence `level`.

    VaR_α  = −Q_{1-α}(r)            [positive number = loss]
    ES_α   = −E[r | r ≤ Q_{1-α}(r)]

    Parameters
    ----------
    returns : 1-D array of log-returns
    level : float in (0, 1), e.g. 0.95

    Returns
    -------
    (var, es) : both expressed as positive loss magnitudes
    """
    r = np.asarray(returns).ravel()
    q = np.percentile(r, (1 - level) * 100)
    var = -q
    es = -float(np.mean(r[r <= q]))
    return float(var), float(es)


def _max_drawdown(log_returns: np.ndarray) -> float:
    """Maximum drawdown from a vector of log-returns (as positive fraction)."""
    cumulative = np.exp(np.cumsum(log_returns / 100.0))  # price index
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (rolling_max - cumulative) / rolling_max
    return float(np.max(drawdown))


def _ljung_box_pval_sq(returns: np.ndarray, lags: int = 20) -> float:
    """Ljung-Box p-value for autocorrelation in squared returns (lag 20)."""
    r2 = np.asarray(returns).ravel() ** 2
    result = acorr_ljungbox(r2, lags=[lags], return_df=True)
    return float(result["lb_pvalue"].iloc[-1])


# ---------------------------------------------------------------------------
# Main validation functions
# ---------------------------------------------------------------------------

def compute_metrics(
    real: pd.Series,
    synthetic_paths: np.ndarray,
) -> dict:
    """
    Compute all validation metrics comparing historical and synthetic returns.

    The synthetic distribution is represented by pooling all paths × all
    time steps, which gives the marginal (unconditional) distribution.
    Per-path statistics (drawdown, Ljung-Box) are computed path-by-path and
    then averaged.

    Parameters
    ----------
    real : pd.Series
        Historical percentage log-returns.
    synthetic_paths : np.ndarray, shape (n_paths, horizon)
        Simulated percentage log-returns from fit_gjr_garch + simulate().

    Returns
    -------
    dict with sub-dicts: moments, quantiles, risk, acf, drawdown
    """
    from scipy import stats as spstats

    r_real = real.dropna().values
    r_synth_pool = synthetic_paths.ravel()

    # --- Moments ---
    moments = {
        "real": {
            "mean": float(np.mean(r_real)),
            "std": float(np.std(r_real, ddof=1)),
            "skewness": float(spstats.skew(r_real)),
            "excess_kurtosis": float(spstats.kurtosis(r_real, fisher=True)),
        },
        "synth": {
            "mean": float(np.mean(r_synth_pool)),
            "std": float(np.std(r_synth_pool, ddof=1)),
            "skewness": float(spstats.skew(r_synth_pool)),
            "excess_kurtosis": float(spstats.kurtosis(r_synth_pool, fisher=True)),
        },
    }

    # --- Tail quantiles ---
    quantile_levels = [0.01, 0.05, 0.95, 0.99]
    quantiles = {
        "levels": quantile_levels,
        "real": [float(np.percentile(r_real, q * 100)) for q in quantile_levels],
        "synth": [float(np.percentile(r_synth_pool, q * 100)) for q in quantile_levels],
    }

    # --- VaR and ES ---
    var95_r, es95_r = compute_var_es(r_real, 0.95)
    var99_r, es99_r = compute_var_es(r_real, 0.99)
    var95_s, es95_s = compute_var_es(r_synth_pool, 0.95)
    var99_s, es99_s = compute_var_es(r_synth_pool, 0.99)

    risk = {
        "real":  {"var95": var95_r, "var99": var99_r, "es95": es95_r, "es99": es99_r},
        "synth": {"var95": var95_s, "var99": var99_s, "es95": es95_s, "es99": es99_s},
    }

    # --- ACF: Ljung-Box on squared returns, fraction-of-rejection approach ---
    # Mean p-value is misleading for short paths (252 obs, lag 20 → low power).
    # Fraction of paths rejecting at 5% is more interpretable: under i.i.d.
    # null only ~5% would reject; a GARCH model should produce >> 40%.
    lb_pvals = [_ljung_box_pval_sq(path) for path in synthetic_paths]
    lb_reject_frac = float(np.mean([p < 0.05 for p in lb_pvals]))
    acf_metrics = {
        "real_lb_pval": _ljung_box_pval_sq(r_real),
        "synth_lb_pval_mean": float(np.mean(lb_pvals)),
        "synth_lb_reject_frac": lb_reject_frac,
        "synth_lb_pval_paths": lb_pvals,
    }

    # --- Drawdown ---
    real_dd = _max_drawdown(r_real)
    synth_dds = [_max_drawdown(path) for path in synthetic_paths]
    drawdown = {
        "real": real_dd,
        "synth_mean": float(np.mean(synth_dds)),
        "synth_p5": float(np.percentile(synth_dds, 5)),
        "synth_p95": float(np.percentile(synth_dds, 95)),
        "synth_all": synth_dds,
    }

    return {
        "moments": moments,
        "quantiles": quantiles,
        "risk": risk,
        "acf": acf_metrics,
        "drawdown": drawdown,
    }


def evaluate_thresholds(metrics: dict) -> list[dict]:
    """
    Apply PASS/FAIL thresholds to computed metrics.

    Returns a list of dicts, one per metric:
        {"metric", "label", "real", "synth", "threshold", "status", "note"}
    """
    m = metrics
    rows = []

    def _rel_err(real_val: float, synth_val: float) -> float:
        return abs(synth_val / real_val - 1.0) if abs(real_val) > 1e-10 else abs(synth_val)

    # Excess kurtosis
    k_r = m["moments"]["real"]["excess_kurtosis"]
    k_s = m["moments"]["synth"]["excess_kurtosis"]
    re = _rel_err(k_r, k_s)
    rows.append({
        "metric": "kurtosis_rel_error",
        "label": THRESHOLD_LABELS["kurtosis_rel_error"],
        "real": round(k_r, 3),
        "synth": round(k_s, 3),
        "error": round(re, 4),
        "threshold": THRESHOLDS["kurtosis_rel_error"],
        "status": "PASS" if re <= THRESHOLDS["kurtosis_rel_error"] else "FAIL",
        "note": "Relative error of excess kurtosis; governs tail mass",
    })

    for level_str, threshold_key in [("var95", "var95_rel_error"), ("var99", "var99_rel_error"),
                                      ("es95", "es95_rel_error"), ("es99", "es99_rel_error")]:
        r_val = m["risk"]["real"][level_str]
        s_val = m["risk"]["synth"][level_str]
        re = _rel_err(r_val, s_val)
        rows.append({
            "metric": threshold_key,
            "label": THRESHOLD_LABELS[threshold_key],
            "real": round(r_val, 4),
            "synth": round(s_val, 4),
            "error": round(re, 4),
            "threshold": THRESHOLDS[threshold_key],
            "status": "PASS" if re <= THRESHOLDS[threshold_key] else "FAIL",
            "note": f"Positive loss magnitude (%); relative error vs historical",
        })

    # Ljung-Box: PASS means fraction of rejecting paths >= threshold
    lb_frac = m["acf"]["synth_lb_reject_frac"]
    rows.append({
        "metric": "lb_reject_frac_min",
        "label": THRESHOLD_LABELS["lb_reject_frac_min"],
        "real": "≈1.00",
        "synth": round(lb_frac, 3),
        "error": round(1.0 - lb_frac, 4),
        "threshold": THRESHOLDS["lb_reject_frac_min"],
        "status": "PASS" if lb_frac >= THRESHOLDS["lb_reject_frac_min"] else "FAIL",
        "note": "Fraction of 252-day paths rejecting no-autocorrelation (power-adjusted for short horizon)",
    })

    return rows
