"""Tests for the Welch t-test power and sample size functions.

welch_statistical_power uses the exact noncentral t distribution
(noncentral_t_cdf, implemented without scipy); the reference is
scipy.stats.nct with the same Welch-Satterthwaite df, counting both
rejection tails.
"""

import itertools

import numpy as np
import pytest
from scipy import stats

from ab_agent.stats.power_and_sample_size import (
    welch_degrees_of_freedom_from_std,
    welch_required_sample_size_per_variant,
    welch_statistical_power,
)
from ab_agent.stats.welch_test import (
    noncentral_t_cdf,
    student_t_cdf,
    welch_degrees_of_freedom,
)


def exact_welch_power(n, std_a, std_b, delta, alpha):
    df = welch_degrees_of_freedom_from_std(std_a, std_b, n)
    noncentrality = abs(delta) / np.sqrt(std_a**2 / n + std_b**2 / n)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # P(T < -t_crit; nc) = P(T > t_crit; -nc). Written as sf because
    # scipy's nct.cdf returns nan far in the lower tail (e.g. df=198, nc=12).
    return stats.nct.sf(t_crit, df, noncentrality) + stats.nct.sf(t_crit, df, -noncentrality)


# ---------------------------------------------------------------- reference values


@pytest.mark.parametrize(
    "std_a, std_b, delta, expected_n",
    [(1, 1, 1, 17), (1, 1, 0.5, 64), (10, 15, 1, 2552)],
)
def test_reference_sample_sizes(std_a, std_b, delta, expected_n):
    n = welch_required_sample_size_per_variant(std_a, std_b, delta, 0.05, 0.8)
    assert abs(n - expected_n) <= 1


# ---------------------------------------------------------------- building blocks


@pytest.mark.parametrize("df", [1, 2.5, 5, 10, 30, 100, 1000])
@pytest.mark.parametrize("t", [-6.0, -2.0, -0.5, 0.0, 0.3, 1.96, 4.0])
def test_student_t_cdf_matches_scipy(t, df):
    assert student_t_cdf(t, df) == pytest.approx(stats.t.cdf(t, df), rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("df", [1, 2.5, 5, 17.3, 198, 5000])
@pytest.mark.parametrize("nc", [-12.0, -1.3, 0.0, 0.4, 2.5, 12.0])
@pytest.mark.parametrize("t", [-20.0, -2.0, -0.1, 0.0, 0.5, 1.96, 4.0, 30.0])
def test_noncentral_t_cdf_matches_scipy(t, df, nc):
    # Lower tail via sf with mirrored arguments, where scipy stays finite.
    expected = stats.nct.cdf(t, df, nc) if t >= 0 else stats.nct.sf(-t, df, -nc)
    assert noncentral_t_cdf(t, df, nc) == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("df, nc, t", [(39998, 50, 48), (39998, 50, 51), (2000, 40, 38), (10, 40, 30)])
def test_noncentral_t_cdf_large_noncentrality(df, nc, t):
    # Poisson weights e^{-nc^2/2} underflow here; the sum must start at the mode.
    assert noncentral_t_cdf(t, df, nc) == pytest.approx(stats.nct.cdf(t, df, nc), rel=1e-9)


@pytest.mark.parametrize("std_a, std_b, n", [(1, 1, 10), (1, 3, 25), (10, 15, 100), (0.2, 7, 4)])
def test_df_from_std_matches_df_from_data(std_a, std_b, n):
    # Build samples whose sample standard deviations are exactly std_a and std_b.
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    a = (a - a.mean()) / a.std(ddof=1) * std_a
    b = (b - b.mean()) / b.std(ddof=1) * std_b
    assert welch_degrees_of_freedom_from_std(std_a, std_b, n) == pytest.approx(
        welch_degrees_of_freedom(a, b), rel=1e-12
    )


def test_df_equal_std_is_pooled_df():
    for n in (2, 10, 100):
        assert welch_degrees_of_freedom_from_std(3, 3, n) == pytest.approx(2 * n - 2)


def test_df_bounds():
    # Welch df lies between n - 1 and 2n - 2.
    for std_b in (0.01, 0.5, 1, 2, 100):
        df = welch_degrees_of_freedom_from_std(1, std_b, 50)
        assert 49 <= df <= 98


# ---------------------------------------------------------------- accuracy vs exact


POWER_GRID = list(
    itertools.product(
        [20, 30, 50, 100, 300],  # n per group
        [(1, 1), (1, 2), (3, 1), (10, 15)],  # (std_a, std_b)
        [0.3, 0.5, 0.8, 1.2],  # delta in units of sqrt((std_a^2 + std_b^2) / 2)
        [0.05, 0.01],  # alpha
    )
)


@pytest.mark.parametrize("n, stds, d, alpha", POWER_GRID)
def test_power_close_to_noncentral_t(n, stds, d, alpha):
    std_a, std_b = stds
    delta = d * np.sqrt((std_a**2 + std_b**2) / 2)
    approx = welch_statistical_power(n, std_a, std_b, delta, alpha)
    exact = exact_welch_power(n, std_a, std_b, delta, alpha)
    assert abs(approx - exact) < 0.01


# ---------------------------------------------------------------- inverse consistency

SAMPLE_SIZE_CASES = [
    # (std_a, std_b, delta, alpha, power)
    (1, 1, 1, 0.05, 0.8),
    (1, 1, 0.5, 0.05, 0.8),
    (10, 15, 1, 0.05, 0.8),
    (1, 2, 0.3, 0.01, 0.9),
    (5, 1, -2, 0.05, 0.95),
    (2, 2, 3, 0.1, 0.8),
    (1, 1, 0.1, 0.001, 0.99),
    (1, 1, 5, 0.05, 0.8),  # tiny n, near the n >= 2 floor
]


@pytest.mark.parametrize("std_a, std_b, delta, alpha, power", SAMPLE_SIZE_CASES)
def test_inverse_consistency(std_a, std_b, delta, alpha, power):
    n = welch_required_sample_size_per_variant(std_a, std_b, delta, alpha, power)
    assert n >= 2
    assert welch_statistical_power(n, std_a, std_b, delta, alpha) >= power
    if n > 2:
        assert welch_statistical_power(n - 1, std_a, std_b, delta, alpha) < power


# ---------------------------------------------------------------- monotonicity


def test_larger_delta_needs_smaller_n():
    sizes = [welch_required_sample_size_per_variant(1, 2, d, 0.05, 0.8) for d in (0.1, 0.2, 0.5, 1, 2)]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == len(sizes)


def test_higher_power_needs_larger_n():
    sizes = [welch_required_sample_size_per_variant(1, 2, 0.3, 0.05, p) for p in (0.5, 0.7, 0.8, 0.9, 0.99)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_smaller_alpha_needs_larger_n():
    sizes = [welch_required_sample_size_per_variant(1, 2, 0.3, a, 0.8) for a in (0.2, 0.1, 0.05, 0.01, 0.001)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_larger_std_needs_larger_n():
    sizes = [welch_required_sample_size_per_variant(1, s, 0.5, 0.05, 0.8) for s in (0.5, 1, 2, 4)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


@pytest.mark.parametrize("std_a, std_b, delta, alpha", [(1, 1, 0.5, 0.05), (10, 15, 1, 0.01)])
def test_power_increases_with_n(std_a, std_b, delta, alpha):
    powers = [welch_statistical_power(n, std_a, std_b, delta, alpha) for n in (2, 5, 20, 100, 1000, 20000)]
    # Power never decreases; it rises strictly until it reaches 1.0 in floating point.
    assert all(a <= b for a, b in itertools.pairwise(powers))
    assert all(a < b for a, b in itertools.pairwise(powers) if a < 1.0)
    assert powers[-1] > 0.99


def test_power_symmetric_in_sign_of_delta_and_group_order():
    p = welch_statistical_power(40, 1, 3, 0.7, 0.05)
    assert welch_statistical_power(40, 1, 3, -0.7, 0.05) == p
    assert welch_statistical_power(40, 3, 1, 0.7, 0.05) == pytest.approx(p, rel=1e-12)


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"std_a": 0, "std_b": 1, "delta": 1, "alpha": 0.05, "power": 0.8},
        {"std_a": 1, "std_b": -1, "delta": 1, "alpha": 0.05, "power": 0.8},
        {"std_a": 1, "std_b": 1, "delta": 0, "alpha": 0.05, "power": 0.8},
        {"std_a": 1, "std_b": 1, "delta": 1, "alpha": 0, "power": 0.8},
        {"std_a": 1, "std_b": 1, "delta": 1, "alpha": 1.5, "power": 0.8},
        {"std_a": 1, "std_b": 1, "delta": 1, "alpha": 0.05, "power": 1},
    ],
)
def test_sample_size_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        welch_required_sample_size_per_variant(**kwargs)


def test_power_rejects_n_below_two():
    with pytest.raises(ValueError):
        welch_statistical_power(1, 1, 1, 1, 0.05)
