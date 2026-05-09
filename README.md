# Brent Crude Stress-Test Scenario Generator

Synthetic market scenario generation for Brent crude oil, built for financial stress testing and
risk-aware strategy development. Uses **GJR-GARCH(1,1) with Student-t innovations** — chosen
after empirical diagnostics confirmed volatility clustering, an asymmetric leverage effect, and
heavy tails in the historical data.

## Quick start (single command)

```bash
git clone <repo-url> brent-stress-test && cd brent-stress-test && pip install -e . && python run.py
```

That's it. The pipeline:
1. Fetches Brent crude daily close prices (BZ=F, Yahoo Finance, ≥10 years)
2. Runs statistical diagnostics (moments, ACF, QQ, Hill, mean-excess)
3. Fits GJR-GARCH(1,1) with Student-t innovations
4. Simulates 1,000 synthetic price paths (252 trading days each)
5. Validates synthetic vs historical on VaR, ES, ACF, drawdowns
6. Writes **`reports/report.html`** — open in any browser, no server needed

**Python ≥ 3.11 required. Internet access required for data fetch.**

## Optional flags

```bash
python run.py --start 2014-01-01   # extend data history
python run.py --n-paths 5000       # more simulation paths
python run.py --horizon 504        # two-year simulation horizon
python run.py --seed 123           # reproduce with different seed
```

## Run tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

Tests use synthetic data and run without network access.

## Repository structure

```
brent-stress-test/
├── run.py                     # single entry point
├── pyproject.toml             # pinned dependencies
├── CLAUDE.md                  # AI tool context
├── AIUSAGE.md                 # AI-assisted development writeup
├── docs/
│   └── aws-deployment.md      # AWS deployment design
├── src/
│   └── brent_stress/
│       ├── data.py            # yfinance fetch + log-returns
│       ├── diagnostics.py     # EDA: moments, ACF, QQ, Hill, mean-excess
│       ├── model.py           # GJR-GARCH fit + simulate
│       ├── validation.py      # metrics + PASS/FAIL thresholds
│       └── report.py          # self-contained HTML report
├── reports/                   # generated output (git-ignored)
└── tests/
    └── test_smoke.py
```

## What the model reproduces — and what it doesn't

| Property | Reproduced |
|----------|-----------|
| Volatility clustering (ACF of squared returns) | Yes |
| Leverage effect (bad news → more vol) | Yes |
| Heavy tails (excess kurtosis) | Yes |
| Volatility regime shifts (2020, 2022) | **No** |
| Skewed innovations (supply shock asymmetry) | **No** |
| Cross-asset co-movement | **No** |

See `reports/report.html` Section 4 for a detailed failure mode analysis.

## Reproducibility

Given the **same downloaded data** the simulation output is fully deterministic (fixed seed controls
arch's internal `numpy.random.Generator`).  The one source of run-to-run variation is the data
fetch itself: yfinance may return slightly different values for the current day's last bar if
prices are still updating intraday.  The simulation paths, model parameters, and all tail-risk
metrics will be identical if the fetched data is identical.

## Requirements

- Python ≥ 3.11
- Internet access for initial data fetch (Brent crude via Yahoo Finance)
- All other dependencies in `pyproject.toml`
