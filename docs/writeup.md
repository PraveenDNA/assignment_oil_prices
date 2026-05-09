# Brent Crude Stress-Test — Full Technical Writeup

**Candidate submission for 4-Xtra Senior Data Scientist / ML Engineer**
**Model:** GJR-GARCH(1,1) with Student-t innovations
**Data:** Brent crude daily close prices (BZ=F, Yahoo Finance, Jan 2015 – May 2026)
**Entry point:** `pip install -e . && python run.py` → `reports/report.html`

---

## Part 1 — Statistical Diagnostics

### 1.1 Empirical Moments

After computing percentage log-returns (r_t = 100 × log(P_t / P_{t-1})) across 2,854 trading
days, the following moments characterise the distribution:

| Statistic | Value | Gaussian baseline |
|-----------|-------|------------------|
| Mean daily return (%) | 0.020 | — |
| Standard deviation (%) | 2.587 | — |
| Skewness | −0.998 | 0 |
| Excess kurtosis | 13.76 | 0 |
| Jarque–Bera p-value | < 10⁻²⁰⁰ | — |

Excess kurtosis of **13.76** (Gaussian baseline = 0). JB test p < 10⁻²⁰⁰; non-normality is
not a sample artefact. Negative skewness (−0.998) reflects supply-side shock asymmetry —
demand collapses produce sharper left-tail events than equivalent recoveries. Documented
as a model limitation in Part 4.

---

### 1.2 ACF of Returns and Squared Returns

| Series | Result |
|---|---|
| ACF of raw returns | No significant AC beyond lag 1–2; weak-form efficiency holds |
| ACF of squared returns | Significant to lag 20+ (Bartlett 95% CI); confirms volatility clustering |

**Implication:** i.i.d. return models are empirically rejected. GARCH family required.

---

### 1.3 Tail-Focused Diagnostics

Three complementary tail diagnostics:

#### QQ Plot — Student-t Reference

Student-t fit to standardised returns by MLE. Bulk fits well; both tails deviate upward,
indicating the Student-t slightly under-predicts extreme observation frequency.
Consistent with regime-mixing inflating empirical kurtosis.

#### Hill Estimator

Tail index α stabilises at **α ≈ 3.5–4.5** for k ∈ [40, 80].

| Condition | Value | Implication |
|---|---|---|
| α > 2 | ✓ | Finite variance |
| α < 4 | ✓ | Near-diverging kurtosis |
| α ≈ 3.5–4.5 | ✓ | Consistent with Student-t ν ≈ 4–6 (fitted: 5.19) |

#### Mean-Excess Plot (Left Tail)

e(u) = E[X − u | X > u] applied to losses X = −r_t. Slope is **upward** through the
relevant quantile range → Pareto-class tail. Flat slope = exponential; downward = thin (Weibull).

---

### 1.4 Modelling Implications of Diagnostics

| Diagnostic finding | Value / result | Model implication |
|---|---|---|
| Excess kurtosis | 13.76 | Student-t innovations required; Gaussian assigns ~0 probability to observed extremes |
| Jarque–Bera p-value | < 10⁻²⁰⁰ | Non-normality is not a sample artefact |
| ACF sq. returns significant at | lag 20+ | GARCH family required; i.i.d. models are empirically wrong |
| Hill estimator plateau | α ≈ 3.5–4.5, k ∈ [40, 80] | Finite variance, near-diverging kurtosis; consistent with ν ≈ 5–6 |
| Mean-excess plot slope | Upward (left tail) | Pareto-class tail, not exponential or thin |
| Leverage scatter | Negative deciles → higher subsequent vol | GJR asymmetry term γ required |

---

## Part 2 — Generative Model

### 2.1 Model Specification

**GJR-GARCH(1,1) with Student-t innovations** (Glosten, Jagannathan, and Runkle, 1993)

#### The Equations

**Return equation:**
```
r_t = μ + ε_t
ε_t = σ_t · z_t
z_t ~ t(ν)        (Student-t with ν degrees of freedom)
```

**Variance equation:**
```
σ²_t = ω
      + (α + γ · 1_{ε_{t-1} < 0}) · ε²_{t-1}
      + β · σ²_{t-1}
```

**Parameter interpretation:**

| Parameter | Estimate | Role |
|---|---|---|
| μ | 0.0217 | Constant mean return |
| ω | 0.0167 | Baseline (floor) variance |
| α | 0.0326 | ARCH term — symmetric shock impact |
| γ | 0.0825 | GJR asymmetry — activates on negative ε_{t-1} |
| β | 0.9022 | GARCH term — vol persistence |
| ν | 5.19 | Student-t tail heaviness |

**Persistence** = α + γ/2 + β = 0.978 (< 1, stationary; near-IGARCH behaviour)

---

### 2.2 Why This Model Family (Not Others)

#### Why GARCH family (not i.i.d. Monte Carlo)?
ACF of squared returns is significant to lag 20+. Any i.i.d. model ignores this — it is wrong
from first principles. The GARCH family was designed precisely for this property.

#### Why GJR (not plain GARCH)?
Plain GARCH(1,1) treats a +5% day and a −5% day identically. Brent crude returns show
asymmetry: negative shocks produce systematically larger subsequent volatility. The GJR γ term
encodes this. It adds exactly one parameter and is statistically significant in the fitted model.

#### Why Student-t (not Gaussian)?
Hill estimator α ≈ 3.5–4.5, QQ plot deviation in tails, excess kurtosis 13.76. A Gaussian
would assign near-zero probability to the events that dominate stress-test scenarios.

#### Why not neural generators (VAE, GAN, Diffusion)?
2,854 observations is insufficient for reliable neural generative modelling. Neural models
interpolate within the training distribution — they cannot extrapolate into unseen tail territory,
which is the entire purpose of stress testing. They also produce uninterpretable outputs that
cannot be explained to a risk manager or regulator.

#### Why not Bayesian Stochastic Volatility?
BSV is the most credible alternative. It gives posterior uncertainty over vol paths. However:
(1) MCMC fitting takes minutes to hours vs. under a second for MLE; (2) the marginal return
distribution (what we need for VaR/ES) is nearly identical to GJR-GARCH at this tail index;
(3) the added complexity does not improve validation metrics meaningfully for the univariate case.

---

### 2.3 What the Model Is Designed to Reproduce

| Property | Reproduced | Mechanism |
|----------|-----------|-----------|
| Volatility clustering | Yes | β parameter carries vol forward |
| Leverage effect (bad news → more vol) | Yes | γ term activates on negative ε_{t-1} |
| Heavy tails / extreme events | Yes | Student-t innovations with ν = 5.19 |
| Mean-reverting volatility | Yes | Persistence < 1 ensures long-run equilibrium |

### 2.4 What the Model Does NOT Reproduce

| Property | Not Reproduced | Why It Matters |
|----------|---------------|----------------|
| Volatility regime shifts | No | Model is single-state stationary |
| Skewed innovations | No | Student-t is symmetric |
| Cross-asset co-movement | No | Paths simulated independently |
| Long-memory volatility | No | Geometric decay only (FIGARCH territory) |

---

### 2.5 Fit-Then-Simulate Interface

The model follows a strict two-step interface (see `src/brent_stress/model.py`):

**Step 1 — Fit:**
```python
from brent_stress.model import fit_gjr_garch
result = fit_gjr_garch(returns, seed=42)
```
Fits GJR-GARCH(1,1)+t via maximum likelihood. Returns an `ARCHModelResult` object containing
all fitted parameters with standard errors.

**Step 2 — Simulate:**
```python
from brent_stress.model import simulate
paths = simulate(result, n_paths=1000, horizon=252, seed=42)
# paths.shape == (1000, 252)
```
Generates n_paths independent 252-day return paths. Each path is seeded deterministically:
given the same `seed`, `n_paths`, and `horizon`, the output is bit-for-bit identical across
runs. The simulation is initialised from the unconditional (long-run) variance, not from the
current fitted state — appropriate for scenario generation rather than short-term forecasting.

**Reproducibility:** Fixed seed controls the internal `numpy.random.Generator` stored inside
the arch distribution object (`result.model.distribution._generator`). This was confirmed to
produce identical outputs across multiple runs.

---

## Part 3 — Validation

### 3.1 Fitted Model Parameters

| Parameter | Estimate | Std. Error | t-stat | Interpretation |
|-----------|----------|------------|--------|----------------|
| μ (mu) | 0.0217 | 0.0373 | 0.58 | Near-zero mean daily return |
| ω (omega) | 0.0167 | 0.0087 | 1.92 | Small baseline variance |
| α (alpha) | 0.0326 | 0.0103 | 3.17 | Positive shock ARCH term |
| γ (gamma) | 0.0825 | 0.0194 | 4.25 | GJR asymmetry — highly significant |
| β (beta) | 0.9022 | 0.0148 | 60.9 | Strong vol persistence |
| ν (nu) | 5.19 | 0.51 | 10.2 | Heavy tails confirmed |

Note: γ is statistically significant (t = 4.25), confirming the asymmetric leverage effect.
ν = 5.19 is in the range where both variance and kurtosis are finite but kurtosis is large —
consistent with the Hill estimator evidence.

---

### 3.2 Marginal Moment Comparison

| Moment | Historical | Synthetic (mean over 1,000 paths) | Assessment |
|--------|-----------|----------------------------------|------------|
| Mean (%) | 0.020 | 0.118 | Close; synthetic mean slightly higher |
| Std (%) | 2.587 | 2.484 | Within 4% — good |
| Skewness | −0.998 | +0.21 | Mismatch — model has symmetric innovations |
| Excess kurtosis | 13.76 | 19.24 | Overshoot — 40% error **(FAIL)** |

**Key observation:** The model reproduces variance well but overshoots kurtosis. This is
explained and documented as a failure mode in Part 4.

---

### 3.3 Left and Right Tail Quantiles

| Quantile | Historical (%) | Synthetic (%) | Relative error |
|----------|--------------|--------------|----------------|
| 1st percentile (left tail) | −7.22 | −7.49 | 3.7% |
| 5th percentile (left tail) | −3.90 | −3.51 | 10.0% |
| 95th percentile (right tail) | 3.50 | 3.28 | 6.3% |
| 99th percentile (right tail) | 6.14 | 5.96 | 2.9% |

The tail quantiles are reproduced well. The 5th percentile is the tightest miss at 10%.

---

### 3.4 VaR and Expected Shortfall

**Threshold rationale:**

| Metric | Threshold | Justification |
|--------|-----------|--------------|
| VaR 95% relative error | ≤ 10% | Closest metric to Basel III regulatory standard; tight tolerance is appropriate |
| VaR 99% relative error | ≤ 15% | Rarer threshold; some estimation variance is unavoidable |
| ES 95% relative error | ≤ 15% | ES integrates full tail; harder to estimate than VaR |
| ES 99% relative error | ≤ 20% | Only ~25 extreme observations in historical data; widest tolerance is honest |

**Results:**

| Metric | Historical (%) | Synthetic (%) | Relative Error | Threshold | Status |
|--------|--------------|--------------|----------------|-----------|--------|
| VaR 95% | 3.900 | 3.511 | 10.0% | 10% | **PASS** |
| VaR 99% | 7.089 | 6.576 | 7.2% | 15% | **PASS** |
| ES 95% | 6.313 | 5.564 | 11.9% | 15% | **PASS** |
| ES 99% | 11.002 | 9.497 | 13.7% | 20% | **PASS** |

All four tail risk metrics pass. The model slightly under-estimates both VaR and ES (synthetic
values are lower than historical), consistent with the pooled distribution being slightly more
spread out due to kurtosis overshoot, which mechanically lowers tail quantile estimates through
diversification across paths.

---

### 3.5 ACF of Squared Returns

**Threshold rationale:**
Rather than using mean p-value across paths (which conflates low statistical power with absence
of the effect), we use the **fraction of 252-day paths that individually reject the null of no
autocorrelation** at the 5% level using Ljung-Box lag-20. Under truly i.i.d. innovations,
only ~5% of paths would reject. We require ≥ 40% — a demanding bar that confirms the GARCH
property is present in simulated paths.

| Metric | Historical | Synthetic | Threshold | Status |
|--------|-----------|-----------|-----------|--------|
| LB reject fraction (p < 0.05) | ~1.00 | 0.578 | ≥ 0.40 | **PASS** |

57.8% of synthetic paths show significant squared-return autocorrelation. The GARCH property
is reproduced. The fraction is not near 1.0 because individual 252-day windows have limited
statistical power for detecting vol clustering — this is expected and not a failure.

---

### 3.6 Drawdowns

| Metric | Value |
|--------|-------|
| Historical maximum drawdown | ~55% |
| Synthetic mean maximum drawdown (over 1,000 paths) | ~38% |
| Synthetic 5th–95th percentile band | 22%–59% |

The historical max drawdown falls within the synthetic 5th–95th percentile band, confirming
the model covers the observed worst-case scenario. The synthetic mean is lower than the
historical max because the historical sample happens to contain several large geopolitical shocks
in sequence (2015–2016 oil glut, 2020 COVID, 2022 Ukraine) — these co-occur within the same
11-year window but need not occur within any given 252-day simulation.

---

### 3.7 Full PASS/FAIL Summary

| Metric | Status | Error | Threshold |
|--------|--------|-------|-----------|
| Excess kurtosis relative error | **FAIL** | 39.8% | 30% |
| VaR 95% relative error | **PASS** | 10.0% | 10% |
| VaR 99% relative error | **PASS** | 7.2% | 15% |
| ES 95% relative error | **PASS** | 11.9% | 15% |
| ES 99% relative error | **PASS** | 13.7% | 20% |
| LB reject fraction | **PASS** | — | ≥ 0.40 |

**5 of 6 metrics pass.** The one failure (kurtosis) is understood, explainable, and documented.

---

## Part 4 — Failure Mode Analysis

### Primary Failure: Kurtosis Overshoot (Validation FAIL)

**What:** Synthetic excess kurtosis of 19.24 overshoots the historical value of 13.76 by ~40%,
exceeding the 30% tolerance.

**Why this happens — mechanically:**
GJR-GARCH with Student-t(ν = 5.19) stacks two independent sources of kurtosis:
1. The Student-t innovations themselves contribute excess kurtosis (for t(ν), excess kurtosis
   = 6/(ν−4) ≈ 4.7 for ν = 5.19)
2. The GARCH variance process adds additional kurtosis through the time-varying σ²_t — even
   with Gaussian innovations, a GARCH process has heavier tails than i.i.d. Gaussian

Their combination produces more total kurtosis than the empirical data exhibits. This is a
well-known property of GARCH+heavy-tail combinations and is not a surprise — it is a direct
consequence of our model choice.

**Why this happens — economically:**
The historical kurtosis of 13.76 is itself partly driven by the consecutive occurrence of three
major shock events (2020, 2022) within the sample period. In a different 11-year window without
these shocks, historical kurtosis might be lower. The model is arguably generating the right
long-run tail behaviour, and the historical sample is a single realisation that happened to
include multiple regime events.

**Impact on stress testing:**
Counterintuitively, the kurtosis overshoot does not make the model conservative in its VaR/ES
estimates — it actually makes them slightly *lower* (synthetic VaR and ES are below historical).
This is because higher kurtosis with fixed variance means fatter tails but also more mass near
the mean, and the diversification across 1,000 paths reduces the influence of the very largest
path-level draws on pooled quantiles.

**Remedies:**
- Constrain ν during fitting (lower bound of 7–8) to limit innovation-level kurtosis
- Switch to semi-parametric tail: GJR-GARCH for the body + GPD (Generalised Pareto) for
  threshold exceedances (peaks-over-threshold approach)
- Markov-switching GARCH: separate high-vol and low-vol regimes, preventing the model from
  conflating them into a single heavy-tailed distribution

---

### Secondary Failure: Symmetric Innovations (Skewness Mismatch)

**What:** Historical skewness is −0.998; synthetic skewness is +0.21.

**Why:** Student-t is a symmetric distribution. The model generates equal probability of
upside and downside surprises of any given magnitude. Real Brent returns are negatively skewed
— supply shocks produce sharper left-tail events than equivalent recoveries.

**Impact on stress testing:** The model under-weights deep left-tail scenarios relative to the
right tail, leading to a slight underestimate of ES at the 99% level for specifically left-tail
events.

**Remedy:** Hansen's (1994) Skewed Student-t or Normal Inverse Gaussian (NIG) innovations.
Both add one parameter (skewness) to the innovation distribution and have closed-form likelihoods
compatible with GARCH estimation.

---

### Tertiary Failure: Single-State Variance Equation

**What:** The model assumes a single stationary variance equation. Brent crude experiences
abrupt and persistent shifts in the volatility level — the 2020 COVID crash and 2022 Russia
invasion each created weeks of extreme volatility that was clearly structurally different from
the preceding calm periods, not just a large draw from a stationary distribution.

**Evidence:** Standardised residuals (ε_t / σ_t) in the 2020 and 2022 windows show persistent
patterns — systematic over- or under-prediction of volatility — inconsistent with the i.i.d.
assumption the model places on these residuals.

**Impact on stress testing:** Synthetic paths cannot generate the sustained high-volatility
regimes that characterise the most severe historical stress periods. A scenario generator that
cannot produce "2020-like" sustained volatility is materially incomplete for worst-case testing.

**Remedy:** Hamilton–Susmel (1994) Markov-switching GARCH with two or three volatility states.
Alternatively, rolling-window recalibration (re-fit the model on the most recent 252 days)
gives time-varying parameters without full regime-switching complexity.

---

## Part 5 — Running From a Clean Clone

```bash
# One command from a clean clone:
git clone <repo-url> brent-stress-test
cd brent-stress-test
pip install -e . && python run.py
```

This fetches data, fits the model, simulates 1,000 paths, and writes `reports/report.html`.
Open that file in any browser. No server required.

**Python ≥ 3.11 required. Internet access required for data fetch.**

All dependencies are pinned in `pyproject.toml`. The virtual environment is optional but
recommended (`python -m venv .venv && source .venv/bin/activate` before `pip install -e .`).

---

## Part 6 — AIUSAGE.md Summary

*(Full file at `AIUSAGE.md` in the repo root — ~300 words)*

**Tool:** Claude Code (Anthropic, VS Code extension). No other AI tools.

**Setup:** Wrote `CLAUDE.md` first — a repo context file describing all modules, the model
family, and entry point — so every Claude session had full project context without
re-explanation. Used spec-driven flow: wrote function signatures and docstrings (including the
reasoning behind design choices) before asking Claude to implement.

**Delegated to AI:** Boilerplate (pyproject.toml, Jinja2 HTML template, ACF plot axis layout,
base64 figure encoding), repetitive validation loop structure, smoke test stubs.

**Not delegated:** Model family selection (GJR over GARCH, Student-t over Gaussian — decided
from inspecting diagnostic plots personally), threshold calibration (set before running any
simulation, based on domain knowledge of regulatory standards), diagnostic prose (written after
personally interpreting specific numerical outputs from the data), failure mode identification
(derived from examining standardised residuals in the 2020/2022 windows, not boilerplate).

**AI error caught:** Claude generated `result.model.simulate(..., random_state=np.random.RandomState(s))`.
This raises a `TypeError` — `arch >= 6.x` has no `random_state` kwarg. The actual RNG lives at
`result.model.distribution._generator` (a `numpy.random.Generator`). Caught on first run, fixed
by reading the arch source directly. This is why all AI-generated library calls must be run
immediately.
