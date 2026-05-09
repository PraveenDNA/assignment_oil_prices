# Brent Crude Oil — Stress-Test Scenario Engine

**Model:** GJR-GARCH(1,1) + Student-t innovations
**Data:** BZ=F (Brent crude futures), Yahoo Finance, Jan 2015 – present
**Assessment:** 4-Xtra Senior Data Scientist / ML Engineer take-home

---

## Single command — from a clean clone

```bash
git clone git@github.com:PraveenDNA/assignment_oil_prices.git brent-stress-test
cd brent-stress-test
pip install -e .
python run.py
```

Opens `reports/report.html` in any browser. No server required.

**Requirements:** Python ≥ 3.11 · Internet access (yfinance fetches Brent crude on first run)

---

## All scripts and outputs

| Script | Output | Description |
|---|---|---|
| `python run.py` | `reports/report.html` | Main pipeline: fit → 1 000-path simulation → validate → HTML report |
| `python eda.py` | `reports/eda_report.html` | 11-section EDA with annotated events, tail diagnostics, leverage analysis |
| `python model_selection.py` | `reports/model_selection.html` | 6-model GARCH comparison scored on AIC/BIC, Kupiec POF, simulation accuracy |
| `python scenarios.py` | `reports/scenario_analysis.docx` | 10 global macro/geopolitical stress scenarios → fan charts + risk metrics |
| `python nl_scenario.py` | `reports/nl-<name>-<date>.docx` | Natural language input → Claude extracts params → model → LLM briefing note |

### Recommended run order

```bash
pip install -e .

python eda.py                 # 1. inspect the data
python model_selection.py     # 2. empirical model selection
python run.py                 # 3. main pipeline + validation
python scenarios.py           # 4. global stress scenarios

# NL engine requires Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
python nl_scenario.py --query "What if OPEC collapses and floods the market?"
```

---

## Dependencies

Exact versions pinned in `pyproject.toml`. Full transitive freeze in `requirements.txt` (59 packages).

```
yfinance==1.3.0   arch==8.0.0       pandas==3.0.2    numpy==2.4.4
scipy==1.17.1     matplotlib==3.10.9 statsmodels==0.14.6  Jinja2==3.1.6
python-docx==1.2.0  anthropic==0.100.0
```

For strict reproducibility:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

---

## Model

```
r_t   = μ + ε_t
ε_t   = σ_t · z_t,    z_t ~ t(ν)
σ²_t  = ω + (α + γ · 1_{ε_{t-1}<0}) · ε²_{t-1} + β · σ²_{t-1}

Persistence  α + γ/2 + β  ≈  0.978
Fitted ν                  ≈  5.19
```

**Model selection rationale** (empirical, not assumed):

| Evidence from EDA | Model implication |
|---|---|
| ACF of squared returns significant to lag 20+ | GARCH family required |
| Leverage scatter: negative deciles amplify subsequent vol more than positive | GJR over plain GARCH (γ > 0, t-stat 4.25) |
| Hill estimator plateau α ≈ 3.5–4.5 | Student-t over Gaussian (ν ≈ 5.19) |
| PACF of squared returns: no significant partial AC beyond lag 1 | Order (1,1) sufficient |
| 2,854 observations | Rules out neural generators (VAE/GAN) — insufficient for reliable tail extrapolation |

---

## Validation summary (1 000 paths × 252 days)

| Metric | Historical | Synthetic | Error | Threshold | Status |
|---|---|---|---|---|---|
| Excess kurtosis | 13.76 | 19.24 | 39.8% | ≤ 30% | **FAIL** |
| VaR 95% | 3.900% | 3.511% | 10.0% | ≤ 10% | PASS |
| VaR 99% | 7.089% | 6.576% | 7.2% | ≤ 15% | PASS |
| ES 95% | 6.313% | 5.564% | 11.9% | ≤ 15% | PASS |
| ES 99% | 11.002% | 9.497% | 13.7% | ≤ 20% | PASS |
| LB reject fraction | ~1.00 | 0.578 | — | ≥ 0.40 | PASS |

Kurtosis overshoot is a known GARCH+Student-t interaction: model stacks GARCH-induced kurtosis
on top of innovation kurtosis. VaR/ES metrics all pass. See `docs/writeup.md` §4 for remedies.

---

## What the model reproduces / does not

| Property | Status | Evidence |
|---|---|---|
| Volatility clustering | ✓ | ACF sq. returns significant to lag 20+; LB reject frac 57.8% |
| Leverage effect | ✓ | γ = 0.0825, t-stat 4.25 |
| Heavy tails | ✓ | ν = 5.19; VaR99/ES99 within threshold |
| Volatility regime shifts (2020, 2022) | ✗ | Standardised residuals show persistent misfit in crisis windows |
| Skewed innovations | ✗ | Student-t symmetric; historical skew −0.998, synthetic +0.21 |
| Cross-asset co-movement | ✗ | Paths independent by construction |

---

## Analyst contributions vs AI-generated code

| Component | Origin |
|---|---|
| Model selection decision path (GJR vs GARCH vs EGARCH; Student-t vs Gaussian) | Analyst — derived from EDA diagnostics |
| Threshold calibration (`THRESHOLDS` in `validation.py`) | Analyst — set before any simulation ran; Basel III / estimation-noise rationale |
| Diagnostic interpretation (specific numbers: kurtosis 13.76, Hill α 3.5–4.5, ACF lag 20) | Analyst — written after reading actual output |
| Failure mode identification (kurtosis overshoot mechanism, three remedies) | Analyst — traced through model mechanics |
| 10 global scenario parameter design (`scenarios.py`) | Analyst — each parameter tied to named historical event |
| NL engine two-stage Claude pipeline architecture | Analyst — design decision; includes input validation layer |
| pyproject.toml, Jinja2 template, base64 encoding, Word XML helpers, smoke test stubs | AI-generated scaffolding |

See `AIUSAGE.md` for the full AI usage disclosure. See `docs/writeup.md` for the complete technical writeup.

---

## Reproducibility

Given identical fetched data, all outputs are deterministic. `SEED=42` derives per-path seeds
via `numpy.random.default_rng`, injected directly into `result.model.distribution._generator`
(arch's internal RNG). Path k is identical regardless of total `n_paths`.

---

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

27 smoke tests. Run without network access (synthetic data). Cover: fit shapes, persistence < 1,
simulate reproducibility, VaR/ES relationship, PASS/FAIL evaluation.

---

## Repository structure

```
brent-stress-test/
├── run.py                      # main pipeline
├── eda.py                      # exploratory data analysis (11 sections)
├── model_selection.py          # 6-model GARCH comparison
├── scenarios.py                # 10 global stress scenarios → DOCX
├── nl_scenario.py              # natural language → GJR-GARCH → LLM briefing
├── pyproject.toml              # exact pinned dependencies
├── requirements.txt            # full transitive freeze
├── AIUSAGE.md                  # AI usage disclosure
├── CLAUDE.md                   # AI tool context file
├── docs/
│   ├── writeup.md              # full technical writeup (6 sections)
│   └── aws-deployment.md       # full-stack AWS architecture
├── src/brent_stress/
│   ├── data.py                 # yfinance fetch + log-returns
│   ├── diagnostics.py          # moments, ACF, QQ, Hill, mean-excess
│   ├── model.py                # GJR-GARCH fit + simulate
│   ├── validation.py           # VaR/ES/Kupiec + PASS/FAIL
│   └── report.py               # Jinja2 HTML report
├── reports/                    # generated outputs (committed)
└── tests/test_smoke.py
```
