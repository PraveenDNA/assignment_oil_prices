"""
Smoke tests — fast, network-free, verifies shapes and basic invariants.

All tests use a synthetic return Series generated from a known RNG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_returns() -> pd.Series:
    """
    Synthetic GARCH-like returns for testing.
    Real-looking: heavier tails than Gaussian, mild volatility clustering.
    No network access required.
    """
    rng = np.random.default_rng(42)
    n = 600
    # Generate simple GARCH(1,1)-like variance process
    omega, alpha, beta = 0.05, 0.08, 0.88
    sig2 = np.zeros(n)
    sig2[0] = omega / (1 - alpha - beta)
    eps = rng.standard_t(df=8, size=n)
    returns = np.zeros(n)
    for t in range(1, n):
        sig2[t] = omega + alpha * (returns[t - 1] ** 2) + beta * sig2[t - 1]
        returns[t] = np.sqrt(sig2[t]) * eps[t]

    return pd.Series(
        returns,
        index=pd.date_range("2019-01-01", periods=n, freq="B"),
        name="log_return",
    )


@pytest.fixture(scope="session")
def fitted_model(sample_returns):
    """Fit GJR-GARCH on the synthetic returns once per session."""
    from brent_stress.model import fit_gjr_garch
    return fit_gjr_garch(sample_returns, seed=42)


@pytest.fixture(scope="session")
def synthetic_paths(fitted_model):
    """Small simulation (50 paths × 60 days) for fast tests."""
    from brent_stress.model import simulate
    return simulate(fitted_model, n_paths=50, horizon=60, seed=42)


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------

class TestMoments:
    def test_compute_moments_keys(self, sample_returns):
        from brent_stress.diagnostics import compute_moments
        m = compute_moments(sample_returns)
        expected = {"n", "mean", "std", "skewness", "excess_kurtosis", "jb_stat", "jb_pvalue"}
        assert expected.issubset(m.keys())

    def test_compute_moments_types(self, sample_returns):
        from brent_stress.diagnostics import compute_moments
        m = compute_moments(sample_returns)
        assert isinstance(m["n"], int)
        assert all(isinstance(m[k], float) for k in ["mean", "std", "skewness",
                                                       "excess_kurtosis", "jb_stat", "jb_pvalue"])

    def test_n_matches_input(self, sample_returns):
        from brent_stress.diagnostics import compute_moments
        m = compute_moments(sample_returns)
        assert m["n"] == len(sample_returns)

    def test_std_positive(self, sample_returns):
        from brent_stress.diagnostics import compute_moments
        m = compute_moments(sample_returns)
        assert m["std"] > 0


# ---------------------------------------------------------------------------
# diagnostics.py
# ---------------------------------------------------------------------------

class TestHillEstimator:
    def test_keys(self, sample_returns):
        from brent_stress.diagnostics import hill_estimator
        h = hill_estimator(sample_returns)
        assert "k" in h and "alpha" in h

    def test_lengths_match(self, sample_returns):
        from brent_stress.diagnostics import hill_estimator
        h = hill_estimator(sample_returns)
        assert len(h["k"]) == len(h["alpha"])

    def test_alphas_positive(self, sample_returns):
        from brent_stress.diagnostics import hill_estimator
        h = hill_estimator(sample_returns)
        alphas = [a for a in h["alpha"] if not np.isnan(a)]
        assert all(a > 0 for a in alphas)


class TestPlots:
    """Verify plot functions return Figure objects without errors."""

    def test_plot_return_series(self, sample_returns):
        import matplotlib.pyplot as plt
        from brent_stress.diagnostics import plot_return_series
        df = pd.DataFrame({"Close": np.exp(sample_returns.cumsum() / 100) * 80,
                           "log_return": sample_returns}, index=sample_returns.index)
        fig = plot_return_series(df)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_acf(self, sample_returns):
        import matplotlib.pyplot as plt
        from brent_stress.diagnostics import plot_acf_diagnostics
        fig = plot_acf_diagnostics(sample_returns)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_qq(self, sample_returns):
        import matplotlib.pyplot as plt
        from brent_stress.diagnostics import plot_qq
        fig = plot_qq(sample_returns)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_hill(self, sample_returns):
        import matplotlib.pyplot as plt
        from brent_stress.diagnostics import hill_estimator, plot_hill
        fig = plot_hill(hill_estimator(sample_returns))
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_mean_excess(self, sample_returns):
        import matplotlib.pyplot as plt
        from brent_stress.diagnostics import plot_mean_excess
        fig = plot_mean_excess(sample_returns)
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------

class TestFitGJRGARCH:
    def test_fit_returns_result(self, fitted_model):
        from arch.univariate.base import ARCHModelResult
        assert isinstance(fitted_model, ARCHModelResult)

    def test_param_names_present(self, fitted_model):
        """GJR-GARCH(1,1)+t must have omega, alpha[1], beta[1], nu at minimum."""
        names = set(fitted_model.params.index)
        assert "omega" in names
        assert "alpha[1]" in names
        assert "beta[1]" in names
        assert "nu" in names

    def test_persistence_below_one(self, fitted_model):
        from brent_stress.model import persistence
        p = persistence(fitted_model)
        assert p < 1.0, f"Persistence {p:.4f} ≥ 1 — non-stationary model"

    def test_nu_reasonable(self, fitted_model):
        nu = float(fitted_model.params.get("nu", fitted_model.params.get("Nu", 10.0)))
        assert 2.0 < nu < 100.0, f"Implausible degrees of freedom: {nu}"


class TestSimulate:
    def test_shape(self, synthetic_paths):
        assert synthetic_paths.shape == (50, 60)

    def test_no_nan(self, synthetic_paths):
        assert not np.isnan(synthetic_paths).any()

    def test_no_inf(self, synthetic_paths):
        assert np.isfinite(synthetic_paths).all()

    def test_reproducible(self, fitted_model):
        from brent_stress.model import simulate
        a = simulate(fitted_model, n_paths=10, horizon=30, seed=7)
        b = simulate(fitted_model, n_paths=10, horizon=30, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, fitted_model):
        from brent_stress.model import simulate
        a = simulate(fitted_model, n_paths=10, horizon=30, seed=7)
        b = simulate(fitted_model, n_paths=10, horizon=30, seed=8)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# validation.py
# ---------------------------------------------------------------------------

class TestValidation:
    def test_compute_metrics_keys(self, sample_returns, synthetic_paths):
        from brent_stress.validation import compute_metrics
        m = compute_metrics(sample_returns, synthetic_paths)
        assert "moments" in m
        assert "risk" in m
        assert "acf" in m
        assert "drawdown" in m

    def test_var_es_positive(self, sample_returns, synthetic_paths):
        from brent_stress.validation import compute_metrics
        m = compute_metrics(sample_returns, synthetic_paths)
        assert m["risk"]["real"]["var95"] > 0
        assert m["risk"]["real"]["es95"] > m["risk"]["real"]["var95"]

    def test_evaluate_thresholds_returns_list(self, sample_returns, synthetic_paths):
        from brent_stress.validation import compute_metrics, evaluate_thresholds
        m = compute_metrics(sample_returns, synthetic_paths)
        table = evaluate_thresholds(m)
        assert isinstance(table, list)
        assert len(table) > 0

    def test_threshold_rows_have_required_keys(self, sample_returns, synthetic_paths):
        from brent_stress.validation import compute_metrics, evaluate_thresholds
        m = compute_metrics(sample_returns, synthetic_paths)
        table = evaluate_thresholds(m)
        for row in table:
            assert "status" in row
            assert row["status"] in ("PASS", "FAIL")
            assert "label" in row
            assert "threshold" in row

    def test_at_least_one_pass(self, sample_returns, synthetic_paths):
        """At minimum the Ljung-Box test should PASS on GARCH-generated data."""
        from brent_stress.validation import compute_metrics, evaluate_thresholds
        m = compute_metrics(sample_returns, synthetic_paths)
        table = evaluate_thresholds(m)
        statuses = [r["status"] for r in table]
        assert "PASS" in statuses, f"All metrics FAIL: {table}"


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

class TestVarEs:
    def test_var_es_relationship(self):
        """ES must always be ≥ VaR (by definition)."""
        from brent_stress.validation import compute_var_es
        rng = np.random.default_rng(0)
        r = rng.standard_normal(500)
        var95, es95 = compute_var_es(r, 0.95)
        var99, es99 = compute_var_es(r, 0.99)
        assert es95 >= var95
        assert es99 >= var99
        assert var99 >= var95
