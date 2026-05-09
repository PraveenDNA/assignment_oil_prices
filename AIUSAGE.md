# AI-Assisted Development

## Tools used

**Claude Code** (Anthropic, VS Code extension) was the primary AI tool throughout this project.
No other AI assistants were used.

## Project setup for AI

Before writing any code I created `CLAUDE.md` — a concise repo context file describing all
modules, the chosen model family, and the single entry point. This gave Claude full project
context at the start of every session without re-explanation. I then used a **spec-driven flow**:
I wrote each function's signature and docstring — including the reasoning behind every design
choice — before asking Claude to fill in the implementation. The rationale in `model.py` (why
GJR over plain GARCH, why Student-t over Gaussian) was written by me before any code was
generated, ensuring Claude's output reflected my intent rather than a generic interpretation.

## What I delegated

`pyproject.toml` and dependency management; the Jinja2 HTML report template skeleton;
ACF/PACF plot axis layout; base64 figure encoding in `report.py`; smoke test stubs;
Word document XML cell-shading helpers; yfinance boilerplate.
These are mechanical tasks where AI adds speed with no intellectual cost.

## What I deliberately did not delegate

**Model selection.** Choosing GJR-GARCH over plain GARCH required personally inspecting the
leverage-effect scatter plot (negative return deciles vs subsequent realised vol) and confirming
statistically that the asymmetry term γ is significant in Brent data. Choosing Student-t over
Gaussian was driven by the Hill estimator plateau at α ≈ 3.5–4.5 — I read that number off the
Hill plot and knew it implied near-diverging fourth moment before touching the model code.

**Threshold calibration.** Every value in `THRESHOLDS` in `validation.py` was written and
justified before any simulation ran. The 10% tolerance on VaR95 is not arbitrary — it maps
to ±$0.50 on a $5 daily loss estimate, which is the precision a Basel III internal model
audit would accept. The 20% tolerance on ES99 reflects that 99% ES integrates only ~25
historical observations out of 2,500 days; tighter would be false precision.

**Diagnostic prose.** The three-paragraph EDA interpretation in the HTML report was written
after running the scripts and reading specific output values — actual kurtosis, specific ACF
lag counts, Hill plateau range. It is not a template.

**Failure mode identification.** The kurtosis overshoot (synthetic 19.2 vs historical 13.8)
was identified by inspecting the validation table and tracing it back to the known
GJR-GARCH property of stacking model-induced kurtosis on top of Student-t innovation kurtosis.
Three remedies were proposed in order of parsimony — not from a generic list.

**Scenario design.** The 10 global scenarios in `scenarios.py` required mapping real events
to specific parameter combinations. COVID-19 at 8% starting daily vol matches the realised
peak during the WTI negative-price episode; the −8 bps/day drift matches the demand
destruction pace from the IEA's month-on-month demand revision history.

**Natural language engine design.** The two-stage Claude pipeline in `nl_scenario.py` — extract
parameters from natural language, run model, interpret numbers back into plain English — was my
architectural decision. The key judgement was that stage 1 output must be type-coerced and
range-validated before entering the model to prevent nonsensical inputs from propagating silently.

## One thing AI got wrong

Claude generated `result.model.simulate(..., random_state=np.random.RandomState(s))`, assuming
`arch >= 6.x` follows scikit-learn's `random_state` convention. It does not. The call raised a
`TypeError` on first run. Fixed by reading the arch package source: the distribution object stores
its own `numpy.random.Generator` at `result.model.distribution._generator`, which must be seeded
directly. This illustrates why every AI-generated library call must be executed immediately —
AI tools carry stale knowledge of specific package APIs.
