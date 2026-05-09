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

Boilerplate: `pyproject.toml`, the Jinja2 HTML report template skeleton, ACF plot axis layout,
base64 figure encoding in `report.py`, and smoke test stubs. These are mechanical tasks where
AI adds speed with no loss of intellectual content.

## What I deliberately did not delegate

**Model selection:** Choosing GJR-GARCH over plain GARCH — made after personally inspecting the
ACF of squared returns and confirming leverage asymmetry in the data. **Threshold calibration:**
Every number in `THRESHOLDS` was set before running any simulation, grounded in risk management
practice (Basel III proximity for VaR95, estimation noise budget for ES99). **Diagnostic prose:**
Written after personally interpreting specific numerical outputs (exact kurtosis, Hill α
plateau, ACF lag). **Failure mode identification:** Derived from examining standardised residuals
in the 2020 and 2022 volatility windows — not copied from a boilerplate limitations list.

## One thing AI got wrong

Claude generated `result.model.simulate(..., random_state=np.random.RandomState(s))`, assuming
`arch >= 6.x` follows scikit-learn's `random_state` convention. It does not — there is no such
keyword argument. The call raised a `TypeError` on the first test run. I fixed it by reading
the arch source: the distribution object stores its own `numpy.random.Generator` at
`result.model.distribution._generator`, which must be seeded directly. The fix took two minutes
from error to resolution. This illustrates why every AI-generated library call must be executed
immediately — AI tools carry stale knowledge of specific package APIs.
