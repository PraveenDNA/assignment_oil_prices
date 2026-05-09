# AI-Assisted Development

## Tool

**Claude Code** (Anthropic, VS Code extension). No other AI tools were used.

## Workflow

`CLAUDE.md` was written before any code — a repo context file specifying all modules, the model
family, and the entry point. Every session started with this context so Claude did not need
re-explanation. Function signatures and docstrings (including design rationale) were written
before asking Claude to implement, ensuring outputs reflected specified intent.

## Delegated to AI

`pyproject.toml` structure; Jinja2 HTML report template skeleton; ACF/PACF plot axis layout;
base64 figure encoding in `report.py`; Word document XML cell-shading helpers; smoke test stubs;
yfinance fetch boilerplate; CLI argument parsing.

## Not delegated

**Model selection:** GJR-GARCH chosen over plain GARCH after inspecting leverage-effect scatter
(negative return deciles vs subsequent realised vol confirmed γ significance). Student-t chosen
over Gaussian after Hill estimator placed tail index α ≈ 3.5–4.5, implying near-diverging fourth
moment. Order (1,1) confirmed by PACF of squared returns showing no significant partial AC beyond
lag 1.

**Threshold calibration:** All values in `THRESHOLDS` (`validation.py`) were set before running
any simulation. `var95_rel_error ≤ 0.10` maps to ±$0.50 on a $5 daily loss estimate — the
precision a Basel III internal model audit accepts. `es99_rel_error ≤ 0.20` reflects that 99% ES
integrates ~25 historical observations from 2,500 days; tighter tolerance would be false precision.

**Diagnostic interpretation:** Written after running the EDA and reading specific output values —
exact kurtosis (13.76), Hill plateau range (α ≈ 3.5–4.5 for k ∈ [40, 80]), ACF lag count (20+).

**Failure mode identification:** Kurtosis overshoot (synthetic 19.24 vs historical 13.76) traced
to GJR-GARCH stacking GARCH-induced kurtosis on top of Student-t innovation kurtosis. Three
remedies proposed in order of parsimony: ν lower-bound constraint, GPD tail splice,
Markov-switching GARCH.

**Scenario design:** All 10 global scenarios in `scenarios.py` — parameter combinations (initial
vol, γ multiplier, ν override, drift, horizon) tied to named historical events with documented
rationale. COVID-19 initial vol = 8%/day matched to realised peak during WTI negative-price episode.

**NL engine architecture:** Two-stage Claude pipeline in `nl_scenario.py` — parameter extraction
then result interpretation. Input validation layer added to type-coerce and range-check extracted
JSON before it enters the model.

## AI error caught

Claude generated `result.model.simulate(..., random_state=np.random.RandomState(s))`.
`arch >= 6.x` has no `random_state` kwarg — `TypeError` on first run. Correct approach: seed
`result.model.distribution._generator` (a `numpy.random.Generator`) directly. Fixed by reading
the arch source. Documented in `model.py` with inline comment.
