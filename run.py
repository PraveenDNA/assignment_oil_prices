#!/usr/bin/env python3
"""
Brent Crude Stress-Test Pipeline
=================================

Fetches Brent crude data → runs statistical diagnostics → fits GJR-GARCH(1,1)
with Student-t innovations → simulates synthetic paths → validates against
historical metrics → writes a self-contained HTML report.

Usage
-----
    python run.py
    python run.py --start 2014-01-01 --n-paths 2000 --horizon 504 --seed 99

Output
------
    reports/report.html   (self-contained; open in any browser)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def _banner(msg: str) -> None:
    print(f"\033[1;34m{msg}\033[0m", flush=True)


def _step(n: int, total: int, msg: str) -> float:
    print(f"  [{n}/{total}] {msg} ...", end=" ", flush=True)
    return time.perf_counter()


def _done(t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"\033[32mdone\033[0m  ({elapsed:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Brent crude GJR-GARCH stress-test pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", default="2015-01-01",
                        help="Data start date (ISO format).  Must give ≥10 years.")
    parser.add_argument("--n-paths", type=int, default=1000,
                        help="Number of Monte Carlo paths to simulate.")
    parser.add_argument("--horizon", type=int, default=252,
                        help="Simulation horizon in trading days (252 ≈ 1 year).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed — ensures full reproducibility.")
    args = parser.parse_args()

    Path("reports").mkdir(exist_ok=True)
    TOTAL_STEPS = 5

    _banner("\nBrent Crude Stress-Test Pipeline")
    print(f"  start={args.start}  n_paths={args.n_paths}  horizon={args.horizon}  seed={args.seed}\n")

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    t = _step(1, TOTAL_STEPS, f"Fetching Brent crude (BZ=F) from {args.start}")
    from brent_stress.data import fetch_brent, load_returns
    df = fetch_brent(start=args.start)
    returns = df["log_return"].dropna()
    _done(t)
    print(f"      {len(returns):,} trading days  "
          f"({returns.index.min().date()} → {returns.index.max().date()})")

    # ------------------------------------------------------------------
    # 2. Statistical diagnostics
    # ------------------------------------------------------------------
    t = _step(2, TOTAL_STEPS, "Computing diagnostics (moments, ACF, QQ, Hill, mean-excess)")
    from brent_stress.diagnostics import (
        compute_moments, hill_estimator,
        plot_return_series, plot_acf_diagnostics, plot_qq,
        plot_mean_excess, plot_hill,
    )
    from brent_stress.report import _fig_to_b64

    moments = compute_moments(returns)
    hill = hill_estimator(returns)

    diag_figs = {
        "return_series": _fig_to_b64(plot_return_series(df)),
        "acf":           _fig_to_b64(plot_acf_diagnostics(returns)),
        "qq":            _fig_to_b64(plot_qq(returns)),
        "hill":          _fig_to_b64(plot_hill(hill)),
        "mean_excess":   _fig_to_b64(plot_mean_excess(returns)),
    }
    _done(t)
    print(f"      excess kurtosis = {moments['excess_kurtosis']:.2f}  "
          f"skewness = {moments['skewness']:.3f}  "
          f"JB p = {moments['jb_pvalue']:.2e}")

    # ------------------------------------------------------------------
    # 3. Fit GJR-GARCH(1,1) + Student-t
    # ------------------------------------------------------------------
    t = _step(3, TOTAL_STEPS, "Fitting GJR-GARCH(1,1) with Student-t innovations")
    from brent_stress.model import fit_gjr_garch, persistence as _persistence
    result = fit_gjr_garch(returns, seed=args.seed)
    persist = _persistence(result)
    nu = float(result.params.get("nu", result.params.get("Nu", 10.0)))
    _done(t)
    print(f"      persistence = {persist:.4f}  nu = {nu:.2f}  "
          f"log-likelihood = {result.loglikelihood:.1f}")

    # ------------------------------------------------------------------
    # 4. Simulate
    # ------------------------------------------------------------------
    t = _step(4, TOTAL_STEPS,
              f"Simulating {args.n_paths:,} paths × {args.horizon} days")
    from brent_stress.model import simulate
    synthetic = simulate(result, n_paths=args.n_paths, horizon=args.horizon, seed=args.seed)
    _done(t)

    # ------------------------------------------------------------------
    # 5. Validate and generate report
    # ------------------------------------------------------------------
    t = _step(5, TOTAL_STEPS, "Validating and generating HTML report")
    from brent_stress.validation import compute_metrics, evaluate_thresholds
    from brent_stress.report import generate_report

    metrics = compute_metrics(returns, synthetic)
    table = evaluate_thresholds(metrics)

    output = generate_report(
        df=df,
        returns=returns,
        moments=moments,
        hill_result=hill,
        diagnostic_figs=diag_figs,
        model_result=result,
        model_persistence=persist,
        synthetic_paths=synthetic,
        metrics=metrics,
        validation_table=table,
        n_paths=args.n_paths,
        horizon=args.horizon,
        seed=args.seed,
        output_path=Path("reports/report.html"),
    )
    _done(t)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pass = sum(1 for r in table if r["status"] == "PASS")
    n_total = len(table)
    color = "\033[32m" if n_pass == n_total else "\033[33m"

    print(f"\n  Report → \033[1m{output}\033[0m")
    print(f"  Validation: {color}{n_pass}/{n_total} PASS\033[0m")
    if n_pass < n_total:
        fails = [r["label"] for r in table if r["status"] == "FAIL"]
        print(f"  FAIL: {', '.join(fails)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
