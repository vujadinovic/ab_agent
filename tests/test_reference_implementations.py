"""Compare ab_agent.stats against SciPy and statsmodels reference implementations."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from ab_agent.stats.normal_dist import standard_normal_cdf, standard_normal_ppf
from ab_agent.stats.power_and_sample_size import (
    two_proportion_statistical_power,
    welch_statistical_power,
)
from ab_agent.stats.two_prop_hyp_test import two_prop_z_test
from ab_agent.stats.welch_test import welch_t_test

ALPHAS = [0.2, 0.1, 0.05, 0.01, 0.001, 1e-4, 1e-6]


# ---------------------------------------------------------------- normal distribution


@pytest.mark.parametrize("alpha", ALPHAS)
def test_ppf_critical_values_match_scipy(alpha):
    for p in (alpha / 2, 1 - alpha / 2, alpha, 1 - alpha):
        assert standard_normal_ppf(p) == pytest.approx(stats.norm.ppf(p), rel=1e-14)


@pytest.mark.parametrize("power", [0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
def test_ppf_power_quantiles_match_scipy(power):
    assert standard_normal_ppf(power) == pytest.approx(stats.norm.ppf(power), rel=1e-14, abs=1e-16)


def test_cdf_matches_scipy_on_grid():
    z = np.linspace(-6, 6, 1201)
    np.testing.assert_allclose(standard_normal_cdf(z), stats.norm.cdf(z), rtol=1e-9, atol=0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_cdf_at_critical_values_matches_scipy(alpha):
    z = stats.norm.ppf(alpha / 2)
    assert standard_normal_cdf(z) == pytest.approx(stats.norm.cdf(z), rel=1e-9)
    assert standard_normal_cdf(-z) == pytest.approx(stats.norm.cdf(-z), rel=1e-12)


def test_cdf_absolute_error_everywhere():
    z = np.linspace(-40, 40, 8001)
    np.testing.assert_allclose(standard_normal_cdf(z), stats.norm.cdf(z), rtol=0, atol=1e-15)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_cdf_ppf_roundtrip(alpha):
    for p in (alpha / 2, 1 - alpha / 2):
        assert standard_normal_cdf(standard_normal_ppf(p)) == pytest.approx(p, rel=1e-9)


# ---------------------------------------------------------------- Welch t-test


def _welch_samples():
    rng = np.random.default_rng(12345)
    return [
        (rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)),
        (rng.normal(10, 2, 15), rng.normal(11, 5, 40)),
        (rng.normal(0, 10, 200), rng.normal(1, 15, 180)),
        (rng.exponential(2, 50), rng.exponential(2.5, 70)),
        (rng.normal(0, 1, 3), rng.normal(3, 0.1, 4)),
        ([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0, 10.0]),
    ]


@pytest.mark.parametrize("a, b", _welch_samples())
def test_welch_t_test_matches_scipy(a, b):
    ours = welch_t_test(a, b, confidence_level=0.95)
    # welch_t_test reports mean(b) - mean(a), so pass b first to scipy.
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert ours["t"] == pytest.approx(ref.statistic, rel=1e-12)
    assert ours["df"] == pytest.approx(ref.df, rel=1e-12)
    assert ours["p_value"] == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-15)
    ci = ref.confidence_interval(0.95)
    assert ours["ci_lower"] == pytest.approx(ci.low, rel=1e-9)
    assert ours["ci_upper"] == pytest.approx(ci.high, rel=1e-9)


# ---------------------------------------------------------------- two-proportion z-test


@pytest.mark.parametrize(
    "successes_a, total_a, successes_b, total_b",
    [
        (100, 1000, 120, 1000),
        (384, 3841, 461, 3841),
        (50, 500, 45, 700),
        (3, 40, 9, 35),
        (5000, 100000, 5200, 100000),
        (100, 1000, 100, 1000),
    ],
)
def test_two_prop_z_test_matches_statsmodels(successes_a, total_a, successes_b, total_b):
    ours = two_prop_z_test(successes_a, total_a, successes_b, total_b, 0.95)
    z, p = proportions_ztest([successes_b, successes_a], [total_b, total_a])
    assert ours["z"] == pytest.approx(z, rel=1e-12, abs=1e-15)
    assert ours["p_value"] == pytest.approx(p, rel=1e-9)


# ---------------------------------------------------------------- power vs statsmodels


@pytest.mark.parametrize("n", [10, 20, 50, 100, 500])
@pytest.mark.parametrize("d", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("alpha", [0.05, 0.01])
@pytest.mark.parametrize("std", [1.0, 7.0])
def test_welch_power_equal_std_vs_statsmodels(n, d, alpha, std):
    # With std_a == std_b the Welch df is exactly 2n - 2, the df statsmodels uses,
    # so both compute the same exact noncentral t power.
    ours = welch_statistical_power(n, std, std, d * std, alpha)
    ref = TTestIndPower().power(effect_size=d, nobs1=n, alpha=alpha, ratio=1.0, alternative="two-sided")
    if np.isnan(ref):
        pytest.xfail("statsmodels/scipy nctdtr (Boost) returns nan in the deep lower tail; series does not converge")
    assert abs(ours - ref) < 0.02


@pytest.mark.parametrize(
    "n, baseline, effect",
    [
        (3841, 0.10, 0.02),
        (1000, 0.10, 0.02),
        (500, 0.30, 0.05),
        (2000, 0.50, -0.03),
        (10000, 0.05, 0.005),
        (300, 0.20, 0.08),
        (20000, 0.02, 0.003),
    ],
)
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_two_proportion_power_vs_statsmodels_cohens_h(n, baseline, effect, alpha):
    # statsmodels works on Cohen's h (arcsine transform), so only rough agreement.
    ours = two_proportion_statistical_power(n, baseline, effect, alpha)
    h = proportion_effectsize(baseline + effect, baseline)
    ref = NormalIndPower().power(effect_size=h, nobs1=n, alpha=alpha, ratio=1.0, alternative="two-sided")
    assert abs(ours - ref) < 0.03
