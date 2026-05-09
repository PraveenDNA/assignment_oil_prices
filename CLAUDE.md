# Brent Crude Stress-Test — Claude Context

## Purpose
Generates statistically plausible synthetic Brent crude price scenarios for financial stress testing.
Uses GJR-GARCH(1,1) with Student-t innovations, motivated by empirical evidence of volatility
clustering, leverage effect, and heavy tails in the data.

## Key modules (src/brent_stress/)
| Module | Responsibility |
|--------|---------------|
| `data.py` | Fetch BZ=F from yfinance, compute log-returns |
| `diagnostics.py` | Empirical moments, ACF/PACF, QQ plot, Hill estimator, mean-excess |
| `model.py` | GJR-GARCH fit via `arch` package + multi-path simulation |
| `validation.py` | Metrics (VaR, ES, ACF, drawdown) and PASS/FAIL evaluation |
| `report.py` | Self-contained HTML report with embedded base64 figures |

## Entry point
```bash
python run.py                         # full pipeline → reports/report.html
python run.py --n-paths 500 --seed 7  # override defaults
```

## Tests
```bash
python -m pytest tests/ -v
```
Tests use synthetic data (no network); they verify shapes, no-NaN, and threshold evaluation.

## Fixed seed
All stochastic operations use `seed=42` by default. Two runs of `python run.py` must produce
identical `reports/report.html`.

## Model family rationale (do not change without updating model.py docstring and report prose)
The model family was chosen after inspecting:
1. ACF of squared returns → confirmed volatility clustering → GARCH family
2. Leverage asymmetry in residuals → negative shocks raise vol more → GJR over plain GARCH
3. QQ plot + Hill estimator → heavy, near-Pareto tails → Student-t over Gaussian innovations
