"""Monte Carlo checks of the power functions and of the tests' type I error.

Each case simulates N_EXPERIMENTS experiments with a fixed seed. Under H1 the
rejection rate should match the computed power; under H0 it should match alpha.
Marked slow: run with `pytest -m slow`.
"""

import math

import numpy as np
import pytest

from ab_agent.stats.power_and_sample_size import (
    two_proportion_required_sample_size_per_variant,
    two_proportion_statistical_power,
    welch_required_sample_size_per_variant,
    welch_statistical_power,
)
from ab_agent.stats.two_prop_hyp_test import two_prop_z_test
from ab_agent.stats.welch_test import student_t_two_sided_p_value

pytestmark = pytest.mark.slow

N_EXPERIMENTS = 10_000
POWER_TOLERANCE = 0.015


def alpha_tolerance(alpha):
    # About 4 Monte Carlo standard errors of a rejection rate equal to alpha.
    return 4 * math.sqrt(alpha * (1 - alpha) / N_EXPERIMENTS)


# ---------------------------------------------------------------- proportions


def proportion_rejection_rate(rate_a, rate_b, n, alpha, seed):
    rng = np.random.default_rng(seed)
    successes_a = rng.binomial(n, rate_a, N_EXPERIMENTS)
    successes_b = rng.binomial(n, rate_b, N_EXPERIMENTS)
    rejections = sum(
        two_prop_z_test(int(sa), n, int(sb), n, 1 - alpha)["p_value"] < alpha
        for sa, sb in zip(successes_a, successes_b)
    )
    return rejections / N_EXPERIMENTS


@pytest.mark.parametrize(
    "baseline, mde, alpha, power, seed",
    [
        (0.10, 0.02, 0.05, 0.80, 1),
        (0.30, -0.05, 0.05, 0.90, 2),
        (0.50, 0.05, 0.01, 0.80, 3),
        (0.05, 0.02, 0.05, 0.80, 4),
    ],
)
def test_proportions_rejection_rate_under_h1_matches_power(baseline, mde, alpha, power, seed):
    n = two_proportion_required_sample_size_per_variant(baseline, mde, alpha, power)
    expected = two_proportion_statistical_power(n, baseline, mde, alpha)
    observed = proportion_rejection_rate(baseline, baseline + mde, n, alpha, seed)
    assert abs(observed - expected) < POWER_TOLERANCE


@pytest.mark.parametrize(
    "rate, n, alpha, seed",
    [
        (0.10, 3841, 0.05, 11),
        (0.30, 1000, 0.01, 12),
        (0.05, 500, 0.05, 13),
    ],
)
def test_proportions_false_positive_rate_under_h0(rate, n, alpha, seed):
    observed = proportion_rejection_rate(rate, rate, n, alpha, seed)
    assert abs(observed - alpha) < alpha_tolerance(alpha)


# ---------------------------------------------------------------- Welch


def welch_rejection_rate(std_a, std_b, delta, n, alpha, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, std_a, (N_EXPERIMENTS, n))
    b = rng.normal(delta, std_b, (N_EXPERIMENTS, n))
    # Vectorized Welch statistic and df (same formulas as welch_t_test, which is
    # too slow to call 10k times because of its CI bisection); the p-value comes
    # from the package's own t distribution.
    var_a = a.var(axis=1, ddof=1) / n
    var_b = b.var(axis=1, ddof=1) / n
    t = (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt(var_a + var_b)
    df = (var_a + var_b) ** 2 / (var_a**2 / (n - 1) + var_b**2 / (n - 1))
    rejections = sum(student_t_two_sided_p_value(ti, dfi) < alpha for ti, dfi in zip(t, df))
    return rejections / N_EXPERIMENTS


@pytest.mark.parametrize(
    "std_a, std_b, delta, alpha, power, seed",
    [
        (1, 1, 1, 0.05, 0.80, 21),
        (1, 1, 0.5, 0.05, 0.80, 22),
        (1, 3, 1, 0.05, 0.80, 23),
        (10, 15, 3, 0.05, 0.90, 24),
        (2, 1, 1, 0.01, 0.80, 25),
    ],
)
def test_welch_rejection_rate_under_h1_matches_power(std_a, std_b, delta, alpha, power, seed):
    n = welch_required_sample_size_per_variant(std_a, std_b, delta, alpha, power)
    expected = welch_statistical_power(n, std_a, std_b, delta, alpha)
    observed = welch_rejection_rate(std_a, std_b, delta, n, alpha, seed)
    assert abs(observed - expected) < POWER_TOLERANCE


@pytest.mark.parametrize(
    "std_a, std_b, n, alpha, seed",
    [
        (1, 1, 17, 0.05, 31),
        (1, 3, 10, 0.05, 32),
        (10, 15, 50, 0.01, 33),
        (5, 1, 30, 0.05, 34),
    ],
)
def test_welch_false_positive_rate_under_h0(std_a, std_b, n, alpha, seed):
    observed = welch_rejection_rate(std_a, std_b, 0.0, n, alpha, seed)
    assert abs(observed - alpha) < alpha_tolerance(alpha)
