# Brent Crude Oil — Stress-Test Scenario Engine

**GJR-GARCH(1,1) + Student-t** model generating statistically plausible synthetic Brent crude
price scenarios for stress testing. Built for the 4-Xtra Senior Data Scientist / ML Engineer
take-home assessment. Includes a natural language interface where an analyst writes a scenario
in plain English and gets a risk committee briefing note back.

---

## Single command — from a clean clone

```bash
git clone git@github.com-personal:PraveenDNA/assignment_oil_prices.git brent-stress-test
cd brent-stress-test
pip install -e .
python run.py
```

**That is everything.** Open `reports/report.html` in any browser when it finishes.

> **Requirements:** Python ≥ 3.11 · Internet access (yfinance fetches Brent crude data)

---

## All reports and what produces them

Run each script independently; every one writes to `reports/`:

| Command | Output | What it does |
|---|---|---|
| `python run.py` | `reports/report.html` | Full pipeline: fit → simulate → validate → HTML report |
| `python eda.py` | `reports/eda_report.html` | 11-section exploratory data analysis with event annotations |
| `python model_selection.py` | `reports/model_selection.html` | Compares 6 GARCH-family models, scores each on 4 dimensions, selects best |
| `python scenarios.py` | `reports/scenario_analysis.docx` | 10 global macro/geopolitical scenarios → Word document with fan charts |
| `python nl_scenario.py` | `reports/nl-<name>-<date>.docx` | **Natural language** → Claude extracts params → model runs → Claude writes briefing |

### Recommended order for a reviewer

```bash
pip install -e .

# 1. EDA — understand the data before touching the model
python eda.py

# 2. Model selection — see why GJR-GARCH was chosen over 5 alternatives
python model_selection.py

# 3. Main pipeline — fit, simulate, validate, report
python run.py

# 4. Scenario analysis — 10 global stress scenarios
python scenarios.py

# 5. Natural language engine (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python nl_scenario.py --query "What if OPEC collapses and floods the market?"
```

---

## Dependencies — exact pinned versions

`pyproject.toml` pins every dependency to the exact version used during development.
`requirements.txt` contains the full transitive freeze (all 59 packages).

```
yfinance==1.3.0       arch==8.0.0          pandas==3.0.2
numpy==2.4.4          scipy==1.17.1        matplotlib==3.10.9
statsmodels==0.14.6   Jinja2==3.1.6        python-docx==1.2.0
anthropic==0.100.0
```

For strict reproducibility (CI / clean environments):

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

---

## What I built — and what I deliberately thought through myself

This project has a hard separation between **AI-generated scaffolding** and **analyst-authored
intellectual content**. The 30% that is mine is the part that matters for a stress-testing tool:
model selection, calibration, interpretation, and scenario design.

### My ideation (not delegated to AI)

**1. Diagnostic interpretation** — I ran the EDA first and recorded specific empirical findings
before writing any model code. The ACF of squared returns showed significant autocorrelation
out to lag 20+, confirming volatility clustering. The Hill estimator plateau at α ≈ 3.5–4.5
for k ∈ [40, 80] implies finite variance but near-diverging fourth moment — I used this to
rule out Gaussian innovations. The mean-excess plot slope is positive through both tails,
consistent with a Pareto-class tail rather than sub-exponential. These are observations from
actually looking at the numbers, not from a boilerplate limitations list.

**2. Model selection decision path** — I chose GJR-GARCH over plain GARCH because the ACF of
squared returns confirmed clustering AND inspection of the leverage-effect scatter (negative
return deciles vs subsequent realised vol) showed clear asymmetry: negative shocks amplify
future volatility more than positive shocks of equal size. I chose Student-t over Gaussian
because the Hill estimator said so, not because it is conventional. I chose order (1,1) over
higher orders because the PACF of squared returns showed no significant partial autocorrelation
beyond lag 1 — higher orders would overfit. I ran all six GARCH-family alternatives
(`model_selection.py`) and confirmed empirically that GARCH-St wins on AIC/BIC + Kupiec
coverage + simulation accuracy.

**3. Threshold calibration** — Every number in `validation.py:THRESHOLDS` was written before
running any simulation, grounded in risk management practice:
- `var95_rel_error ≤ 0.10`: 95% VaR is a Basel III / FRTB regulatory metric; ±10% tolerance
  means ±$0.50 error on a $5 daily loss estimate — tight but achievable
- `es99_rel_error ≤ 0.20`: Expected Shortfall integrates 25 historical observations at the 99%
  level from 2,500 days of data; 20% honestly reflects estimation noise, not sloppiness
- `lb_reject_frac_min ≥ 0.40`: I switched from mean p-value (misleading on 252-day paths) to
  fraction-of-paths-rejecting after understanding that short horizon Ljung-Box tests have low
  power — the fraction approach is more interpretable and aligns with what GARCH actually claims

**4. Failure mode identification** — The kurtosis overshoot (synthetic 19.2 vs historical 13.8)
is a known GJR-GARCH property: the model stacks GARCH-induced excess kurtosis on top of the
Student-t innovation's kurtosis. I identified three remedies in order of parsimony: (a) skewed
Student-t innovations, (b) variance targeting, (c) Markov-switching GARCH. These came from
reading the residuals, not from a generic limitations section.

**5. Scenario design** — The 10 global scenarios in `scenarios.py` were constructed by mapping
real-world events to specific model parameter combinations. For example, COVID-19 uses 8% daily
starting volatility (matched to the peak realised vol during the WTI negative-price episode),
a negative drift of −8 bps/day (demand destruction pace), and a 63-day horizon (the acute
phase before the OPEC+ emergency cut). These numbers are not arbitrary — each has a documented
rationale in the scenario catalogue.

**6. Natural language engine architecture** — The two-stage Claude pipeline in `nl_scenario.py`
(extraction → simulation → interpretation) was my design choice. The critical insight is that
the LLM output at stage 1 must be structurally validated before entering the model — I added
type coercion and range checks on the extracted JSON to prevent the model from running with
nonsensical parameters (e.g. ν = 0.5 or horizon = 10,000).

### What AI generated

- `pyproject.toml`, Jinja2 HTML template skeleton, base64 figure encoding in `report.py`
- ACF plot axis layout, smoke test stubs, Word document XML formatting
- yfinance data fetch boilerplate, `_b64()` helper, CLI argument parsing

**One thing AI got wrong:** Claude generated `result.model.simulate(..., random_state=np.random.RandomState(s))`,
assuming `arch ≥ 6.x` follows scikit-learn's `random_state` convention. It does not. Fixed by
reading the arch source: the distribution stores its RNG at `result.model.distribution._generator`
(a `numpy.random.Generator`), which must be seeded directly. This is documented in `model.py`
with a comment explaining the fix and why it matters for reproducibility.

---

## Model equations

```
r_t   = μ + ε_t
ε_t   = σ_t · z_t,   z_t ~ t(ν)
σ²_t  = ω + (α + γ · 1_{ε_{t-1}<0}) · ε²_{t-1} + β · σ²_{t-1}

Persistence = α + γ/2 + β  (must be < 1 for stationarity; fitted ≈ 0.978)
```

**Why GJR over plain GARCH:** The leverage term γ captures that negative oil price shocks
amplify future volatility more than positive shocks — confirmed empirically by the leverage
scatter in `eda.py`.

**Why Student-t over Gaussian:** Hill estimator places Brent's tail index α ≈ 3.5–4.5,
implying near-diverging fourth moment. Gaussian innovations dramatically under-weight extreme
scenario probability.

---

## What the model reproduces — and what it doesn't

| Property | Reproduced | Evidence |
|---|---|---|
| Volatility clustering | Yes | ACF of squared returns significant to lag 20+ |
| Leverage effect (bad news → more vol) | Yes | γ > 0, statistically significant |
| Heavy tails | Yes | Student-t ν ≈ 5.2; Kupiec POF test passes at 95% and 99% |
| Volatility regime shifts (2020, 2022) | **No** | Standardised residuals show persistent misfit in crisis windows |
| Skewed innovations | **No** | Student-t is symmetric; Brent shows negative skew ~−0.3 |
| Cross-asset co-movement | **No** | Paths are independent |
| Long-memory volatility | **No** | FIGARCH territory; (1,1) order is sufficient for forecasting |

---

## Repository structure

```
brent-stress-test/
├── run.py                      # main pipeline entry point
├── eda.py                      # 11-section exploratory data analysis
├── model_selection.py          # 6-model GARCH comparison + scoring
├── scenarios.py                # 10 global stress scenarios → DOCX
├── nl_scenario.py              # natural language → model → LLM briefing
├── pyproject.toml              # exact pinned dependencies
├── requirements.txt            # full transitive freeze (59 packages)
├── CLAUDE.md                   # AI tool project context
├── AIUSAGE.md                  # ~350-word AI usage writeup
├── docs/
│   ├── aws-deployment.md       # full-stack AWS architecture design
│   └── writeup.md              # technical writeup covering all 6 sections
├── src/
│   └── brent_stress/
│       ├── data.py             # yfinance fetch + log-returns
│       ├── diagnostics.py      # moments, ACF, QQ, Hill, mean-excess
│       ├── model.py            # GJR-GARCH fit + simulate (with RNG fix)
│       ├── validation.py       # VaR/ES/Kupiec metrics + PASS/FAIL thresholds
│       └── report.py           # self-contained HTML report (Jinja2)
├── reports/                    # generated output (git-ignored; run scripts to produce)
└── tests/
    └── test_smoke.py           # 27 smoke tests; runs without network access
```

---

## Run tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

Tests use synthetic data — no internet required.

---

## Reproducibility

Given the **same fetched data**, all outputs are fully deterministic. The master seed (`SEED=42`)
derives per-path seeds via `numpy.random.default_rng`, which are then injected directly into
`result.model.distribution._generator` (arch's internal RNG). Path k produces identical output
regardless of total `n_paths`. The only source of variation across runs is yfinance returning
a slightly updated intraday bar — model parameters and tail-risk metrics will be identical if
the fetched data is identical.

---

## AWS deployment

See [docs/aws-deployment.md](docs/aws-deployment.md) for the full production architecture:
React SPA + CloudFront → Cognito SSO → API Gateway → Step Functions → ECS Fargate.
Covers invocation path, identity/secrets (three-principal least-privilege), observability
(structured JSON logs + X-Ray + CloudWatch Alarms → PagerDuty), and 100× scaling design.
Estimated cost: ~$19.50/month at 300 runs.
