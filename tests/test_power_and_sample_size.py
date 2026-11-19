"""Tests for the two-proportion power and sample size functions."""

import itertools

import pytest

from ab_agent.stats.power_and_sample_size import (
    two_proportion_required_sample_size_per_variant,
    two_proportion_statistical_power,
)

# (baseline_rate, minimum_detectable_effect, alpha, power)
PARAM_COMBINATIONS = [
    (0.10, 0.02, 0.05, 0.80),
    (0.10, 0.01, 0.05, 0.80),
    (0.05, 0.01, 0.01, 0.90),
    (0.30, 0.05, 0.05, 0.95),
    (0.50, 0.10, 0.10, 0.80),
    (0.20, -0.03, 0.05, 0.80),
    (0.02, 0.005, 0.05, 0.80),
    (0.70, -0.10, 0.001, 0.99),
]


def test_reference_value():
    assert two_proportion_required_sample_size_per_variant(0.10, 0.02, 0.05, 0.8) == 3841


@pytest.mark.parametrize("baseline, mde, alpha, power", PARAM_COMBINATIONS)
def test_inverse_consistency(baseline, mde, alpha, power):
    n = two_proportion_required_sample_size_per_variant(baseline, mde, alpha, power)
    assert two_proportion_statistical_power(n, baseline, mde, alpha) >= power
    assert two_proportion_statistical_power(n - 1, baseline, mde, alpha) < power


@pytest.mark.parametrize("baseline", [0.05, 0.10, 0.30, 0.50])
def test_larger_mde_needs_smaller_n(baseline):
    sizes = [
        two_proportion_required_sample_size_per_variant(baseline, mde, 0.05, 0.8)
        for mde in (0.005, 0.01, 0.02, 0.05)
    ]
    assert sizes == sorted(sizes, reverse=True)
    assert len(set(sizes)) == len(sizes)


@pytest.mark.parametrize("baseline, mde", [(0.10, 0.02), (0.30, -0.05)])
def test_higher_power_needs_larger_n(baseline, mde):
    sizes = [
        two_proportion_required_sample_size_per_variant(baseline, mde, 0.05, power)
        for power in (0.5, 0.7, 0.8, 0.9, 0.99)
    ]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


@pytest.mark.parametrize("baseline, mde", [(0.10, 0.02), (0.30, -0.05)])
def test_smaller_alpha_needs_larger_n(baseline, mde):
    sizes = [
        two_proportion_required_sample_size_per_variant(baseline, mde, alpha, 0.8)
        for alpha in (0.2, 0.1, 0.05, 0.01, 0.001)
    ]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


@pytest.mark.parametrize("baseline, mde, alpha", [(0.10, 0.02, 0.05), (0.50, -0.05, 0.01)])
def test_power_increases_with_n(baseline, mde, alpha):
    powers = [two_proportion_statistical_power(n, baseline, mde, alpha) for n in (50, 100, 500, 1000, 5000, 20000)]
    assert all(a < b for a, b in itertools.pairwise(powers))
    assert powers[-1] > 0.99


# With proportions, power(p, d) != power(p, -d) in general, because the binomial
# variance p(1-p) depends on p. The exact symmetries are relabeling success and
# failure (p -> 1-p, d -> -d) and swapping the two variants (p -> p+d, d -> -d).
@pytest.mark.parametrize("n", [100, 1000, 3841])
@pytest.mark.parametrize("baseline, effect", [(0.10, 0.02), (0.30, 0.05), (0.50, 0.10), (0.05, -0.01)])
def test_symmetry_success_failure_relabeling(n, baseline, effect):
    assert two_proportion_statistical_power(n, baseline, effect, 0.05) == pytest.approx(
        two_proportion_statistical_power(n, 1 - baseline, -effect, 0.05), rel=1e-12
    )


@pytest.mark.parametrize("n", [100, 1000, 3841])
@pytest.mark.parametrize("baseline, effect", [(0.10, 0.02), (0.30, 0.05), (0.50, 0.10), (0.05, -0.01)])
def test_symmetry_variant_swap(n, baseline, effect):
    assert two_proportion_statistical_power(n, baseline, effect, 0.05) == pytest.approx(
        two_proportion_statistical_power(n, baseline + effect, -effect, 0.05), rel=1e-12
    )


@pytest.mark.parametrize("baseline, effect", [(0.10, 0.02), (0.30, 0.05)])
def test_symmetry_sample_size(baseline, effect):
    n = two_proportion_required_sample_size_per_variant(baseline, effect, 0.05, 0.8)
    assert two_proportion_required_sample_size_per_variant(1 - baseline, -effect, 0.05, 0.8) == n
    assert two_proportion_required_sample_size_per_variant(baseline + effect, -effect, 0.05, 0.8) == n
