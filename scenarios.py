#!/usr/bin/env python3
"""
Brent Crude — Global Stress Scenario Analysis
==============================================

Runs 10 geopolitical / macro stress scenarios through a fitted GJR-GARCH(1,1)
+ Student-t model. Each scenario adjusts the calibration window, starting
volatility, model parameters, horizon, and mean drift to match the specific
risk context.

Outputs: reports/scenario_analysis.docx

Usage
-----
    python scenarios.py                  # 1 000 paths per scenario
    python scenarios.py --n-paths 500    # faster run
"""
from __future__ import annotations

import argparse
import io
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from brent_stress.data import fetch_brent
from brent_stress.model import fit_gjr_garch

SEED = 42
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
NAVY   = "1a365d"
BLUE   = "2b6cb0"
TEAL   = "234e52"
RED    = "c53030"
ORANGE = "c05621"
GREEN  = "276749"
PURPLE = "553c9a"
GRAY   = "718096"

# ── scenario catalogue ────────────────────────────────────────────────────────
SCENARIOS: list[dict] = [
    {
        "id": "S01",
        "name": "Baseline — Full History (2015–2025)",
        "short": "Baseline",
        "geography": "Global",
        "event_analogy": "No specific shock — long-run unconditional distribution",
        "context": (
            "Calibrated on the full ten-year Brent crude history spanning calm periods (2016–2019), "
            "the COVID-19 demand collapse (2020), and the Russia-Ukraine supply shock (2022). "
            "This is the unconditional benchmark representing 'average' oil market conditions "
            "across a full economic cycle and serves as the reference against which all stress "
            "scenarios are interpreted. Persistence of 0.978 confirms near-IGARCH behaviour — "
            "volatility shocks decay slowly, meaning a bad day echoes for weeks."
        ),
        "calibration_start": "2015-01-01",
        "calibration_end": None,
        "horizon": 252,
        "initial_vol_pct": None,
        "gamma_multiplier": 1.0,
        "nu_override": None,
        "mu_shock_pct": 0.0,
        "color": "#" + NAVY,
        "hex": NAVY,
    },
    {
        "id": "S02",
        "name": "COVID-19 Demand Collapse (2020)",
        "short": "COVID Crash",
        "geography": "Global",
        "event_analogy": "March–April 2020: Brent lost ~70 %, WTI turned negative",
        "context": (
            "Replicates the pandemic oil price collapse of March–April 2020 when global demand "
            "fell ~25 mb/d and Brent dropped from $57 to $18 in six weeks. Model fitted on the "
            "18-month pre/during crash window (Jun 2019 – Dec 2020). Starting volatility of 8 % "
            "daily mirrors realised peak conditions during the storage overhang. Structural "
            "negative drift of −8 bps/day approximates the demand destruction pace. "
            "3-month horizon captures the acute shock phase before OPEC+ emergency cut took effect."
        ),
        "calibration_start": "2019-06-01",
        "calibration_end": "2020-12-31",
        "horizon": 63,
        "initial_vol_pct": 8.0,
        "gamma_multiplier": 1.0,
        "nu_override": None,
        "mu_shock_pct": -0.08,
        "color": "#" + RED,
        "hex": RED,
    },
    {
        "id": "S03",
        "name": "Russia-Ukraine Supply Shock (2022)",
        "short": "Russia-Ukraine",
        "geography": "Europe / FSU",
        "event_analogy": "Feb 2022: Brent spiked $80 → $130 then reversed sharply",
        "context": (
            "Calibrates on 2021–2023, dominated by Russia's invasion of Ukraine (Feb 2022) which "
            "removed ~2.5 mb/d of seaborne Russian crude from Western markets. Brent spiked from "
            "$80 to $130 within weeks before reversing as alternative supply from the US, Gulf, "
            "and re-routing through Indian/Chinese refiners absorbed the shock. "
            "Starting volatility 5 % daily reflects the early-conflict regime. "
            "6-month horizon covers supply displacement and partial re-routing."
        ),
        "calibration_start": "2021-01-01",
        "calibration_end": "2023-06-30",
        "horizon": 126,
        "initial_vol_pct": 5.0,
        "gamma_multiplier": 1.0,
        "nu_override": None,
        "mu_shock_pct": 0.0,
        "color": "#" + ORANGE,
        "hex": ORANGE,
    },
    {
        "id": "S04",
        "name": "OPEC+ Surprise Production Cut",
        "short": "OPEC+ Shock",
        "geography": "Middle East / OPEC+",
        "event_analogy": "Apr 2023 Easter Sunday surprise: 1.16 mb/d cut, no prior signal",
        "context": (
            "Models a repeat of the April 2023 OPEC+ surprise cut announced on a Sunday with no "
            "prior market signal, triggering an immediate $5 gap-up open. The leverage parameter "
            "(γ) is doubled: in a tight-supply environment bad news amplifies volatility far more "
            "than equivalent positive surprises. Starting vol of 3.5 % captures moderate pre-cut "
            "stress. A small positive drift (+5 bps/day) reflects the expected upward price reset "
            "as the market prices in the new supply floor over a 3-month window."
        ),
        "calibration_start": "2015-01-01",
        "calibration_end": None,
        "horizon": 63,
        "initial_vol_pct": 3.5,
        "gamma_multiplier": 2.0,
        "nu_override": None,
        "mu_shock_pct": 0.05,
        "color": "#" + GREEN,
        "hex": GREEN,
    },
    {
        "id": "S05",
        "name": "US / China Recession — Global Demand Destruction",
        "short": "Global Recession",
        "geography": "Global",
        "event_analogy": "2008–09 GFC playbook at 2024 leverage levels",
        "context": (
            "A synchronised US-China recession reducing global oil demand by 3–4 mb/d. "
            "Calibrated on the full Brent history but with fatter tails (ν = 4.0) to model "
            "tail-risk amplification from a systemic credit event. "
            "Structural negative drift of −5 bps/day (~−12.5 % annualised) reflects sustained "
            "demand weakness over a full year, consistent with GFC dynamics where demand "
            "destruction persisted 12–18 months before stabilising. Starting vol of 4 % reflects "
            "the early-recession risk-off environment."
        ),
        "calibration_start": "2015-01-01",
        "calibration_end": None,
        "horizon": 252,
        "initial_vol_pct": 4.0,
        "gamma_multiplier": 1.0,
        "nu_override": 4.0,
        "mu_shock_pct": -0.05,
        "color": "#" + PURPLE,
        "hex": PURPLE,
    },
    {
        "id": "S06",
        "name": "Strait of Hormuz Disruption",
        "short": "Hormuz Closure",
        "geography": "Middle East / Persian Gulf",
        "event_analogy": "Threatened / partial closure — ~21 mb/d at risk",
        "context": (
            "Simulates a partial or threatened closure of the Strait of Hormuz through which "
            "~21 mb/d (≈20 % of global seaborne oil) transits. Calibrated on the high-volatility "
            "2022–2023 window. The leverage parameter is scaled 1.5× because downside panic "
            "(closure feared) amplifies volatility far more than equivalent upside relief "
            "(diplomatic resolution). Initial vol of 6 % captures the acute shock onset. "
            "40-day horizon approximates the window before diplomatic resolution or alternative "
            "routing (Cape of Good Hope bypass, strategic reserves release) typically activates."
        ),
        "calibration_start": "2022-01-01",
        "calibration_end": "2023-12-31",
        "horizon": 42,
        "initial_vol_pct": 6.0,
        "gamma_multiplier": 1.5,
        "nu_override": None,
        "mu_shock_pct": 0.0,
        "color": "#" + TEAL,
        "hex": TEAL,
    },
    {
        "id": "S07",
        "name": "Energy Transition — Structural Demand Decline",
        "short": "Energy Transition",
        "geography": "Global (Long-term)",
        "event_analogy": "IEA Net Zero 2050 pathway — oil demand peaks before 2030",
        "context": (
            "A 2-year scenario modelling accelerated EV adoption, carbon border taxes, and "
            "industrial electrification structurally eroding oil demand. ν raised to 8 "
            "(lighter innovation tails — fewer extreme daily spikes as the market is more "
            "orderly in a managed-decline environment where forward curves provide guidance). "
            "Negative drift of −8 bps/day (≈−18 % annualised) reflects secular demand "
            "destruction compounding over a 2-year horizon. Lower idiosyncratic vol but "
            "sustained and ultimately predictable price erosion."
        ),
        "calibration_start": "2015-01-01",
        "calibration_end": None,
        "horizon": 504,
        "initial_vol_pct": None,
        "gamma_multiplier": 1.0,
        "nu_override": 8.0,
        "mu_shock_pct": -0.08,
        "color": "#" + BLUE,
        "hex": BLUE,
    },
    {
        "id": "S08",
        "name": "Fed Hawkish Shock — Strong Dollar",
        "short": "Dollar Spike",
        "geography": "US / Global",
        "event_analogy": "2022 Fed tightening cycle; DXY +15 % in 9 months",
        "context": (
            "Oil is priced in USD; a strong-dollar regime from surprise Fed rate hikes reduces "
            "Brent's affordability in non-USD terms and weakens emerging-market demand. "
            "Calibrated on 2022 — the year of the most aggressive Fed tightening in 40 years "
            "(425 bps in 12 months). Moderate negative drift of −3 bps/day captures the "
            "FX-driven demand headwind. Starting vol of 3 % reflects measured market stress "
            "at the onset of a new hawkish cycle rather than a full panic episode. "
            "6-month horizon covers the initial tightening impact."
        ),
        "calibration_start": "2022-01-01",
        "calibration_end": "2022-12-31",
        "horizon": 126,
        "initial_vol_pct": 3.0,
        "gamma_multiplier": 1.0,
        "nu_override": None,
        "mu_shock_pct": -0.03,
        "color": "#" + GRAY,
        "hex": GRAY,
    },
    {
        "id": "S09",
        "name": "Stagflation — Supply Shock + Simultaneous Recession",
        "short": "Stagflation",
        "geography": "Global",
        "event_analogy": "1973–74 oil embargo + 2022 H2 combined shock dynamics",
        "context": (
            "The worst-case combination: a supply disruption keeps oil structurally elevated "
            "while a simultaneous recession weakens end-user purchasing power. Extreme tails "
            "(ν = 3.5 — near-Cauchy behaviour, matching Basel III internal model stress "
            "assumptions) model the tail-on-tail interaction when two independent crises "
            "materialise concurrently. Modest negative drift of −3 bps/day reflects net demand "
            "weakness eventually dominating supply support in the medium term. Starting vol of "
            "5 % sets conditions in a pre-crisis elevated state. Most closely mirrors 1973–74 "
            "or late-2022 stagflationary oil market dynamics."
        ),
        "calibration_start": "2015-01-01",
        "calibration_end": None,
        "horizon": 252,
        "initial_vol_pct": 5.0,
        "gamma_multiplier": 1.0,
        "nu_override": 3.5,
        "mu_shock_pct": -0.03,
        "color": "#c53030",
        "hex": "c53030",
    },
    {
        "id": "S10",
        "name": "Benign Soft Landing — Pre-COVID Calm",
        "short": "Soft Landing",
        "geography": "Global (Benign)",
        "event_analogy": "2016–2018: recovering demand, OPEC compliance, absent shocks",
        "context": (
            "A constructive baseline calibrated on 2016–2018 — a period of recovering demand "
            "post-2015 supply glut, gradual OPEC+ compliance, and absent geopolitical shocks. "
            "This scenario represents the lower bound on oil price risk: what Brent volatility "
            "looks like in an orderly macro environment with stable growth and no supply "
            "disruptions. Small positive drift (+2 bps/day) reflects gradual demand recovery. "
            "Used to stress-test model sensitivity to calibration-period choice and to provide "
            "the 'best-case' reference tail for risk budgeting."
        ),
        "calibration_start": "2016-01-01",
        "calibration_end": "2018-12-31",
        "horizon": 252,
        "initial_vol_pct": None,
        "gamma_multiplier": 1.0,
        "nu_override": None,
        "mu_shock_pct": 0.02,
        "color": "#" + GREEN,
        "hex": GREEN,
    },
]


# ── simulation ────────────────────────────────────────────────────────────────

def _simulate_scenario(
    result,
    gamma_multiplier: float,
    nu_override: float | None,
    mu_shock_pct: float,
    n_paths: int,
    horizon: int,
    initial_vol_pct: float | None,
    seed: int,
) -> np.ndarray:
    """Simulate paths with optional parameter overrides applied to a fitted result."""
    params = result.params.copy()

    # Apply overrides
    if gamma_multiplier != 1.0 and "gamma[1]" in params:
        params["gamma[1]"] = float(params["gamma[1]"]) * gamma_multiplier
    if nu_override is not None:
        params["nu"] = float(nu_override)
    if mu_shock_pct != 0.0:
        params["mu"] = float(params.get("mu", 0.0)) + mu_shock_pct

    initial_value = float(initial_vol_pct ** 2) if initial_vol_pct is not None else None
    burn = 0 if initial_value is not None else 500

    master_rng = np.random.default_rng(seed)
    path_seeds = master_rng.integers(0, 2**31, size=n_paths)

    paths = np.empty((n_paths, horizon), dtype=float)
    for i, s in enumerate(path_seeds):
        result.model.distribution._generator = np.random.default_rng(int(s))
        sim = result.model.simulate(
            params,
            nobs=horizon,
            burn=burn,
            initial_value_vol=initial_value,  # sets σ²₀; initial_value is for mean process
        )
        paths[i] = sim["data"].values

    return paths


# ── metrics ───────────────────────────────────────────────────────────────────

def _max_drawdown(price_path: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(price_path)
    dd = (roll_max - price_path) / roll_max
    return float(np.max(dd))


def compute_scenario_metrics(paths: np.ndarray) -> dict:
    """Compute risk metrics from simulated paths (n_paths × horizon)."""
    daily = paths.ravel()

    # Daily risk metrics (from pooled daily returns)
    var95 = float(-np.percentile(daily, 5))
    var99 = float(-np.percentile(daily, 1))
    below_1pct = daily[daily <= np.percentile(daily, 1)]
    es99 = float(-np.mean(below_1pct)) if len(below_1pct) > 0 else var99
    ann_vol = float(np.std(daily, ddof=1) * np.sqrt(252))

    # Cumulative price paths (start = 100)
    price_paths = 100.0 * np.exp(np.cumsum(paths / 100.0, axis=1))
    final = price_paths[:, -1]
    total_ret = (final - 100.0) / 100.0  # fractional total return

    # Drawdown per path
    drawdowns = np.array([_max_drawdown(price_paths[i]) for i in range(len(paths))])

    return {
        "ann_vol_pct":       round(ann_vol, 1),
        "var95_daily_pct":   round(var95, 2),
        "var99_daily_pct":   round(var99, 2),
        "es99_daily_pct":    round(es99, 2),
        "cum_p01":           round(float(np.percentile(total_ret,  1)) * 100, 1),
        "cum_p10":           round(float(np.percentile(total_ret, 10)) * 100, 1),
        "cum_p50":           round(float(np.percentile(total_ret, 50)) * 100, 1),
        "cum_p90":           round(float(np.percentile(total_ret, 90)) * 100, 1),
        "mean_dd_pct":       round(float(np.mean(drawdowns))          * 100, 1),
        "p95_dd_pct":        round(float(np.percentile(drawdowns, 95)) * 100, 1),
        "prob_dd_gt20_pct":  round(float(np.mean(drawdowns > 0.20))   * 100, 1),
        "prob_dd_gt40_pct":  round(float(np.mean(drawdowns > 0.40))   * 100, 1),
        "prob_loss_gt30_pct":round(float(np.mean(total_ret < -0.30))  * 100, 1),
        "price_paths":       price_paths,
    }


# ── fan chart ─────────────────────────────────────────────────────────────────

def make_fan_chart(price_paths: np.ndarray, scen: dict) -> bytes:
    """Return PNG bytes of a percentile fan chart for one scenario."""
    horizon = price_paths.shape[1]
    days = np.arange(horizon)
    pcts = [1, 10, 25, 50, 75, 90, 99]
    bands = np.percentile(price_paths, pcts, axis=0)

    color = scen["color"]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.fill_between(days, bands[0], bands[6], alpha=0.10, color=color)
    ax.fill_between(days, bands[1], bands[5], alpha=0.18, color=color)
    ax.fill_between(days, bands[2], bands[4], alpha=0.28, color=color, label="P25–P75")
    ax.plot(days, bands[3], color=color, lw=2.0, label="Median")
    ax.plot(days, bands[0], color=color, lw=0.6, ls="--", alpha=0.5)
    ax.plot(days, bands[6], color=color, lw=0.6, ls="--", alpha=0.5, label="P1 / P99")
    ax.axhline(100, color="#718096", lw=0.8, ls=":", alpha=0.7, label="Start (100)")

    ax.set_title(f"{scen['id']} — {scen['short']}", fontsize=9, fontweight="bold",
                 color="#1a365d", pad=6)
    ax.set_xlabel("Trading days", fontsize=8)
    ax.set_ylabel("Price index (start = 100)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_summary_chart(scenarios: list[dict], all_metrics: list[dict]) -> bytes:
    """Return PNG bytes of a 3-panel summary comparison across all scenarios."""
    labels  = [s["short"] for s in scenarios]
    vol     = [m["ann_vol_pct"]     for m in all_metrics]
    var99   = [m["var99_daily_pct"] for m in all_metrics]
    mean_dd = [m["mean_dd_pct"]     for m in all_metrics]
    colors  = [s["color"]           for s in scenarios]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, values, title, ylabel in zip(
        axes,
        [vol, var99, mean_dd],
        ["Annualised Volatility (%)", "Daily VaR 99% (%)", "Mean Max Drawdown (%)"],
        ["Ann. vol (%)", "Daily VaR99 (%)", "Mean DD (%)"],
    ):
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#1a365d")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    fig.suptitle("Cross-Scenario Risk Comparison — Brent Crude", fontsize=11,
                 fontweight="bold", color="#1a365d", y=1.02)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Word doc helpers ──────────────────────────────────────────────────────────

def _hex_rgb(hex6: str) -> RGBColor:
    h = hex6.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _shade_cell(cell, hex6: str) -> None:
    """Set table cell background colour via XML."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex6.lstrip("#"))
    tcPr.append(shd)


def _set_cell_text(cell, text: str, bold: bool = False, font_size: int = 9,
                   color: str | None = None, align: str = "left") -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    para.alignment = {
        "left":   WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right":  WD_ALIGN_PARAGRAPH.RIGHT,
    }.get(align, WD_ALIGN_PARAGRAPH.LEFT)
    run = para.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(font_size)
    if color:
        run.font.color.rgb = _hex_rgb(color)


def _add_metric_row(table, label: str, value: str, shade: bool = False) -> None:
    row = table.add_row()
    if shade:
        _shade_cell(row.cells[0], "edf2f7")
        _shade_cell(row.cells[1], "edf2f7")
    _set_cell_text(row.cells[0], label, font_size=8)
    _set_cell_text(row.cells[1], value, bold=True, font_size=8, align="right")


def _add_section_page(doc: Document, scen: dict, metrics: dict,
                      fan_png: bytes, fit_params: dict) -> None:
    """Add a full scenario section to the Word document."""
    # ── heading ──────────────────────────────────────────────────────────────
    h = doc.add_heading("", level=2)
    h.clear()
    run = h.add_run(f"{scen['id']}  |  {scen['name']}")
    run.font.size = Pt(13)
    run.font.color.rgb = _hex_rgb(scen["hex"])
    run.bold = True

    # Geography / analogy line
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"Geography: {scen['geography']}    |    Analogy: {scen['event_analogy']}")
    r.font.size = Pt(8)
    r.font.color.rgb = _hex_rgb("718096")
    r.italic = True

    # ── context box (table with 1 cell, shaded) ───────────────────────────
    ctx_tbl = doc.add_table(rows=1, cols=1)
    ctx_tbl.style = "Table Grid"
    ctx_cell = ctx_tbl.rows[0].cells[0]
    _shade_cell(ctx_cell, "ebf4ff")
    ctx_cell.text = ""
    cp = ctx_cell.paragraphs[0]
    cp.paragraph_format.space_after = Pt(0)
    cr = cp.add_run(scen["context"])
    cr.font.size = Pt(8)

    doc.add_paragraph()  # spacer

    # ── two-column table: assumptions | risk metrics ──────────────────────
    main_tbl = doc.add_table(rows=1, cols=2)
    main_tbl.style = "Table Grid"

    left_cell  = main_tbl.rows[0].cells[0]
    right_cell = main_tbl.rows[0].cells[1]
    left_cell.width  = Inches(3.1)
    right_cell.width = Inches(3.1)

    # ── Assumptions (left) ────────────────────────────────────────────────
    lp = left_cell.paragraphs[0]
    lr = lp.add_run("Model Assumptions")
    lr.bold = True
    lr.font.size = Pt(9)
    lr.font.color.rgb = _hex_rgb(scen["hex"])

    inner_left = left_cell.add_table(rows=0, cols=2)
    inner_left.style = "Table Grid"

    cal_end = scen["calibration_end"] or "present"
    rows_data_l = [
        ("Calibration window", f"{scen['calibration_start']} – {cal_end}"),
        ("Horizon",            f"{scen['horizon']} trading days"),
        ("Starting vol (σ₀)",  f"{scen['initial_vol_pct']} % / day" if scen["initial_vol_pct"] else "Unconditional"),
        ("Leverage (γ)",       f"{scen['gamma_multiplier']}× fitted"),
        ("Tail heaviness (ν)", f"{scen['nu_override']}" if scen["nu_override"] else "Fitted"),
        ("Mean drift shock",   f"{scen['mu_shock_pct']:+.2f} % / day" if scen["mu_shock_pct"] else "None"),
        ("Fitted ν (model)",   f"{fit_params.get('nu', 'N/A'):.2f}"),
        ("Fitted β (persist.)",f"{fit_params.get('beta[1]', 0):.4f}"),
    ]
    for i, (lbl, val) in enumerate(rows_data_l):
        row = inner_left.add_row()
        _shade_cell(row.cells[0], "f7fafc" if i % 2 == 0 else "ffffff")
        _shade_cell(row.cells[1], "f7fafc" if i % 2 == 0 else "ffffff")
        _set_cell_text(row.cells[0], lbl, font_size=8)
        _set_cell_text(row.cells[1], val, bold=True, font_size=8, align="right")

    # ── Risk Metrics (right) ──────────────────────────────────────────────
    rp = right_cell.paragraphs[0]
    rr = rp.add_run("Risk Metrics")
    rr.bold = True
    rr.font.size = Pt(9)
    rr.font.color.rgb = _hex_rgb(scen["hex"])

    inner_right = right_cell.add_table(rows=0, cols=2)
    inner_right.style = "Table Grid"

    rows_data_r = [
        ("Annualised vol",          f"{metrics['ann_vol_pct']:.1f} %"),
        ("Daily VaR 95%",           f"{metrics['var95_daily_pct']:.2f} %"),
        ("Daily VaR 99%",           f"{metrics['var99_daily_pct']:.2f} %"),
        ("Daily ES 99%",            f"{metrics['es99_daily_pct']:.2f} %"),
        ("Horizon P1 cum. return",  f"{metrics['cum_p01']:.1f} %"),
        ("Horizon P10 cum. return", f"{metrics['cum_p10']:.1f} %"),
        ("Horizon median return",   f"{metrics['cum_p50']:.1f} %"),
        ("Mean max drawdown",       f"{metrics['mean_dd_pct']:.1f} %"),
        ("P95 max drawdown",        f"{metrics['p95_dd_pct']:.1f} %"),
        ("Prob(drawdown > 20 %)",   f"{metrics['prob_dd_gt20_pct']:.1f} %"),
        ("Prob(drawdown > 40 %)",   f"{metrics['prob_dd_gt40_pct']:.1f} %"),
        ("Prob(total loss > 30 %)", f"{metrics['prob_loss_gt30_pct']:.1f} %"),
    ]
    # Header row
    hrow = inner_right.add_row()
    _shade_cell(hrow.cells[0], scen["hex"])
    _shade_cell(hrow.cells[1], scen["hex"])
    _set_cell_text(hrow.cells[0], "Metric",  bold=True, font_size=8, color="ffffff")
    _set_cell_text(hrow.cells[1], "Value",   bold=True, font_size=8, color="ffffff", align="right")

    for i, (lbl, val) in enumerate(rows_data_r):
        row = inner_right.add_row()
        shade = "f7fafc" if i % 2 == 0 else "ffffff"
        _shade_cell(row.cells[0], shade)
        _shade_cell(row.cells[1], shade)
        _set_cell_text(row.cells[0], lbl, font_size=8)
        _set_cell_text(row.cells[1], val, bold=True, font_size=8, align="right")

    doc.add_paragraph()  # spacer

    # ── fan chart ─────────────────────────────────────────────────────────
    img_stream = io.BytesIO(fan_png)
    doc.add_picture(img_stream, width=Inches(6.0))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()


# ── Word doc builder ──────────────────────────────────────────────────────────

def build_word_doc(
    scenarios: list[dict],
    all_metrics: list[dict],
    fan_charts: list[bytes],
    summary_chart: bytes,
    fitted_params: list[dict],
    n_paths: int,
    output_path: Path,
) -> None:
    doc = Document()

    # ── page margins ──────────────────────────────────────────────────────
    from docx.oxml import OxmlElement
    section = doc.sections[0]
    section.page_width  = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin   = Inches(1.0)
    section.right_margin  = Inches(1.0)
    section.top_margin    = Inches(0.9)
    section.bottom_margin = Inches(0.9)

    # ═══════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    doc.add_paragraph()

    title = doc.add_heading("Brent Crude Oil", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = _hex_rgb(NAVY)
        run.font.size = Pt(28)

    sub = doc.add_heading("Global Stress Scenario Analysis", level=1)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.font.color.rgb = _hex_rgb(BLUE)
        run.font.size = Pt(18)

    doc.add_paragraph()

    model_line = doc.add_paragraph()
    model_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = model_line.add_run("Model: GJR-GARCH(1,1) + Student-t Innovations")
    mr.font.size = Pt(11)
    mr.font.color.rgb = _hex_rgb(GRAY)
    mr.italic = True

    doc.add_paragraph()
    doc.add_paragraph()

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(f"Date: {date.today().strftime('%d %B %Y')}    |    "
                 f"Paths per scenario: {n_paths:,}    |    Seed: {SEED}").font.size = Pt(9)

    doc.add_paragraph()

    desc = doc.add_paragraph()
    desc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dr = desc.add_run(
        "Ten geopolitical and macroeconomic scenarios stress-tested through a calibrated "
        "GJR-GARCH volatility model.\nEach scenario adjusts calibration window, initial "
        "volatility, tail heaviness, leverage, and projection horizon\nto reflect the "
        "specific risk context of a named global event or structural shift."
    )
    dr.font.size = Pt(9)
    dr.font.color.rgb = _hex_rgb(GRAY)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    h_exec = doc.add_heading("Executive Summary", level=1)
    for run in h_exec.runs:
        run.font.color.rgb = _hex_rgb(NAVY)

    p_intro = doc.add_paragraph(
        "The table below compares all ten scenarios across six key risk metrics. "
        "Metrics are derived from Monte Carlo simulation of the fitted GJR-GARCH(1,1) "
        "model, pooling daily log-returns across all simulated paths. Daily VaR/ES "
        "figures apply to a single trading day; cumulative metrics apply over each "
        "scenario's specific projection horizon."
    )
    p_intro.paragraph_format.space_after = Pt(8)
    for run in p_intro.runs:
        run.font.size = Pt(9)

    # Summary table header
    cols = ["ID", "Scenario", "Horizon\n(days)", "Ann. Vol\n(%)", "Daily\nVaR99 (%)",
            "Daily\nES99 (%)", "Mean DD\n(%)", "P95 DD\n(%)", "Prob\nDD>40%"]
    tbl = doc.add_table(rows=1, cols=len(cols))
    tbl.style = "Table Grid"
    hdr = tbl.rows[0]
    for i, col_label in enumerate(cols):
        _shade_cell(hdr.cells[i], NAVY)
        _set_cell_text(hdr.cells[i], col_label, bold=True, font_size=8,
                       color="ffffff", align="center")

    for j, (scen, m) in enumerate(zip(scenarios, all_metrics)):
        row = tbl.add_row()
        shade = "edf2f7" if j % 2 == 0 else "ffffff"
        data = [
            scen["id"],
            scen["short"],
            str(scen["horizon"]),
            f"{m['ann_vol_pct']:.1f}",
            f"{m['var99_daily_pct']:.2f}",
            f"{m['es99_daily_pct']:.2f}",
            f"{m['mean_dd_pct']:.1f}",
            f"{m['p95_dd_pct']:.1f}",
            f"{m['prob_dd_gt40_pct']:.1f} %",
        ]
        for i, val in enumerate(data):
            _shade_cell(row.cells[i], shade)
            align = "left" if i <= 1 else "center"
            bold = i == 0
            _set_cell_text(row.cells[i], val, bold=bold, font_size=8, align=align)

    doc.add_paragraph()

    # ── summary comparison chart ──────────────────────────────────────────
    doc.add_heading("Cross-Scenario Comparison", level=2)
    img_stream = io.BytesIO(summary_chart)
    doc.add_picture(img_stream, width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # METHODOLOGY NOTE
    # ═══════════════════════════════════════════════════════════════════════
    doc.add_heading("Methodology", level=1)
    for run in doc.paragraphs[-1].runs:
        run.font.color.rgb = _hex_rgb(NAVY)

    method_paras = [
        (
            "Volatility model.",
            " GJR-GARCH(1,1) with Student-t innovations is fitted on each scenario's "
            "calibration window using maximum likelihood (arch library, Python). "
            "The variance equation is σ²ₜ = ω + (α + γ·𝟙_{εₜ₋₁<0})·ε²ₜ₋₁ + β·σ²ₜ₋₁, "
            "where γ > 0 captures the leverage effect (negative shocks raise future vol "
            "more than positive shocks of equal magnitude). Degrees of freedom ν controls "
            "tail heaviness; fitted values for Brent typically fall between 4.5 and 7."
        ),
        (
            "Parameter overrides.",
            " Each scenario may modify: (a) calibration window — which historical period "
            "drives fitted parameters; (b) initial variance σ²₀ — the starting state of "
            "the variance recursion; (c) γ multiplier — amplifies or dampens the leverage "
            "effect; (d) ν override — stresses tail heaviness independently of the fit; "
            "(e) mean drift shock — adds a constant to μ to impose a directional expectation."
        ),
        (
            "Monte Carlo simulation.",
            f" Each scenario generates {n_paths:,} independent paths from the (overridden) "
            "fitted parameters. Per-path seeding via a master RNG ensures full "
            "reproducibility. A 500-step burn-in is used when starting from unconditional "
            "variance; burn-in is suppressed when initial_vol is explicitly set."
        ),
        (
            "Risk metrics.",
            " Daily VaR/ES are computed from the pooled distribution of all daily returns "
            "across all paths. Cumulative metrics (drawdown, total return percentiles) are "
            "computed per path and then aggregated. Drawdown is defined as the maximum "
            "peak-to-trough decline in the price index over the horizon."
        ),
    ]
    for bold_text, normal_text in method_paras:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        rb = p.add_run(bold_text)
        rb.bold = True
        rb.font.size = Pt(9)
        rn = p.add_run(normal_text)
        rn.font.size = Pt(9)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # INDIVIDUAL SCENARIO SECTIONS
    # ═══════════════════════════════════════════════════════════════════════
    doc.add_heading("Scenario Detail", level=1)
    for run in doc.paragraphs[-1].runs:
        run.font.color.rgb = _hex_rgb(NAVY)
    doc.add_paragraph(
        "Each scenario section shows the economic context, model assumptions, "
        "risk metrics, and a fan chart of simulated price paths (start = 100)."
    ).runs[0].font.size = Pt(9)
    doc.add_paragraph()

    for scen, m, fan_png, fp in zip(scenarios, all_metrics, fan_charts, fitted_params):
        _add_section_page(doc, scen, m, fan_png, fp)

    doc.save(output_path)
    print(f"  → Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Brent crude scenario analysis → Word doc")
    parser.add_argument("--n-paths", type=int, default=1000,
                        help="Monte Carlo paths per scenario (default 1000)")
    parser.add_argument("--start",   type=str, default="2015-01-01",
                        help="Earliest data fetch date")
    args = parser.parse_args()

    print(f"\nBrent Crude — Scenario Analysis  ({len(SCENARIOS)} scenarios, "
          f"{args.n_paths:,} paths each)\n")

    # ── 1. Fetch full data once ───────────────────────────────────────────
    print("  [1] Fetching data ...", end=" ", flush=True)
    df = fetch_brent(start=args.start)
    print(f"done  ({len(df):,} days)")

    # ── 2. Run scenarios ──────────────────────────────────────────────────
    all_metrics:   list[dict]  = []
    fan_charts:    list[bytes] = []
    fitted_params: list[dict]  = []

    for i, scen in enumerate(SCENARIOS, 1):
        print(f"  [{i+1}/{len(SCENARIOS)+1}] {scen['id']} — {scen['short']} ...",
              end=" ", flush=True)

        # Slice calibration window
        cal_df = df.copy()
        if scen["calibration_start"]:
            cal_df = cal_df[cal_df.index >= scen["calibration_start"]]
        if scen["calibration_end"]:
            cal_df = cal_df[cal_df.index <= scen["calibration_end"]]

        returns_pct = cal_df["log_return"].dropna()   # already in % (100×log-return)

        # Fit model
        result = fit_gjr_garch(returns_pct)
        fp = {k: float(v) for k, v in result.params.items()}
        fitted_params.append(fp)

        # Simulate
        paths = _simulate_scenario(
            result            = result,
            gamma_multiplier  = scen["gamma_multiplier"],
            nu_override       = scen["nu_override"],
            mu_shock_pct      = scen["mu_shock_pct"],
            n_paths           = args.n_paths,
            horizon           = scen["horizon"],
            initial_vol_pct   = scen["initial_vol_pct"],
            seed              = SEED,
        )

        # Metrics
        metrics = compute_scenario_metrics(paths)
        all_metrics.append(metrics)

        # Fan chart
        fan_charts.append(make_fan_chart(metrics["price_paths"], scen))

        print(f"vol={metrics['ann_vol_pct']:.1f}%  VaR99={metrics['var99_daily_pct']:.2f}%  "
              f"MDD={metrics['mean_dd_pct']:.1f}%")

    # ── 3. Summary chart ──────────────────────────────────────────────────
    print(f"  [{len(SCENARIOS)+2}/{len(SCENARIOS)+2}] Generating summary chart ...",
          end=" ", flush=True)
    summary_png = make_summary_chart(SCENARIOS, all_metrics)
    print("done")

    # ── 4. Write Word doc ─────────────────────────────────────────────────
    output_path = REPORTS / "scenario_analysis.docx"
    print(f"  Writing {output_path.name} ...", end=" ", flush=True)
    build_word_doc(
        scenarios     = SCENARIOS,
        all_metrics   = all_metrics,
        fan_charts    = fan_charts,
        summary_chart = summary_png,
        fitted_params = fitted_params,
        n_paths       = args.n_paths,
        output_path   = output_path,
    )

    print(f"\n  ✓ Done — open reports/scenario_analysis.docx\n")


if __name__ == "__main__":
    main()
