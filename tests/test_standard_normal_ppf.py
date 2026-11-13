"""Compare standard_normal_ppf against scipy.stats.norm.ppf.

Our implementation uses Algorithm AS 241 (Wichura 1988, accurate to about
1e-16) plus an asymptotic expansion for subnormal p, the same approach as R's
qnorm. SciPy is used as the reference implementation.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from ab_agent.stats.normal_dist import standard_normal_ppf

# Observed max relative difference from scipy is ~1.1e-15 (a few ulp);
# leave some margin for platform-dependent rounding of log/sqrt.
RTOL = 4e-15
# Absolute floor for results very close to 0 (p ~ 0.5).
ATOL = 1e-16

# Region boundaries of the algorithm.
P_CENTRAL_LOW = 0.075  # |p - 0.5| <= 0.425
P_CENTRAL_HIGH = 0.925
P_TAIL_SPLIT = math.exp(-25.0)  # t = sqrt(-log p) = 5,  ~1.39e-11
P_FAR_TAIL_SPLIT = math.exp(-729.0)  # t = 27, ~2.5e-317 (subnormal)


def _assert_close_to_scipy(p):
    ours = standard_normal_ppf(p)
    ref = float(norm.ppf(p))
    assert math.isclose(ours, ref, rel_tol=RTOL, abs_tol=ATOL), (
        f"p={p!r}: ours={ours!r}, scipy={ref!r}, "
        f"abs err={abs(ours - ref):.3e}, rel err={abs(ours - ref) / abs(ref):.3e}"
    )


def _neighbours(x):
    return [math.nextafter(x, 0.0), x, math.nextafter(x, 1.0)]


# --- Typical values used in A/B testing --------------------------------------

COMMON_P = [
    0.5,
    0.8,
    0.9,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.05,
    0.025,
    0.01,
    0.005,
    0.001,
]


@pytest.mark.parametrize("p", COMMON_P)
def test_common_quantiles_match_scipy(p):
    _assert_close_to_scipy(p)


def test_well_known_critical_values():
    assert standard_normal_ppf(0.975) == pytest.approx(1.959963984540054, rel=RTOL)
    assert standard_normal_ppf(0.95) == pytest.approx(1.6448536269514722, rel=RTOL)
    assert standard_normal_ppf(0.8) == pytest.approx(0.8416212335729143, rel=RTOL)


def test_median_is_exactly_zero():
    assert standard_normal_ppf(0.5) == 0.0


# --- Each region of the algorithm -------------------------------------------


@pytest.mark.parametrize(
    "p", np.linspace(P_CENTRAL_LOW, P_CENTRAL_HIGH, 501).tolist()
)
def test_central_region_matches_scipy(p):
    _assert_close_to_scipy(p)


@pytest.mark.parametrize("p", np.geomspace(P_TAIL_SPLIT, P_CENTRAL_LOW, 201).tolist())
def test_tail_region_matches_scipy(p):
    _assert_close_to_scipy(p)


@pytest.mark.parametrize("p", np.geomspace(1e-307, P_TAIL_SPLIT, 201).tolist())
def test_far_tail_region_matches_scipy(p):
    _assert_close_to_scipy(p)


@pytest.mark.parametrize(
    "p",
    [
        5e-324,  # smallest subnormal double
        1e-323,
        1e-320,
        1e-318,
        math.nextafter(P_FAR_TAIL_SPLIT, 0.0),
    ],
)
def test_extreme_tail_subnormal_matches_scipy(p):
    _assert_close_to_scipy(p)


@pytest.mark.parametrize("q", np.geomspace(1e-16, P_CENTRAL_LOW, 151).tolist())
def test_upper_tail_matches_scipy(q):
    # 1 - q is rounded to the nearest double; both functions get the same p.
    _assert_close_to_scipy(1.0 - q)


@pytest.mark.parametrize(
    "p",
    _neighbours(P_CENTRAL_LOW)
    + _neighbours(P_CENTRAL_HIGH)
    + _neighbours(P_TAIL_SPLIT)
    + _neighbours(P_FAR_TAIL_SPLIT)
    + [
        2.2250738585072014e-308,  # smallest normal double
        1e-300,
        math.nextafter(1.0, 0.0),  # largest double < 1
    ],
)
def test_region_boundaries_and_extremes_match_scipy(p):
    _assert_close_to_scipy(p)


# --- Dense sweep with a summary of the error --------------------------------


def test_dense_sweep_max_relative_error():
    rng = np.random.default_rng(12345)
    ps = np.concatenate(
        [
            rng.uniform(0.0, 1.0, 200_000),
            10.0 ** rng.uniform(-323.3, -1, 50_000),
            1.0 - 10.0 ** rng.uniform(-16, -1, 50_000),
        ]
    )
    ps = ps[(ps > 0) & (ps < 1)]

    ours = standard_normal_ppf(ps)
    ref = norm.ppf(ps)

    abs_err = np.abs(ours - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), ATOL)
    worst = int(np.argmax(rel_err))
    print(
        f"\nmax rel err = {rel_err[worst]:.3e} at p={ps[worst]!r}; "
        f"max abs err = {abs_err.max():.3e}; median rel err = {np.median(rel_err):.3e}"
    )
    assert rel_err[worst] <= RTOL, f"worst p={ps[worst]!r}"


# --- Structural properties ---------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 6, 10, 20, 30, 40, 50])
def test_symmetry(k):
    # For p = 2**-k, 1 - p is exactly representable, so ppf(1 - p) == -ppf(p).
    p = 2.0**-k
    assert standard_normal_ppf(1.0 - p) == pytest.approx(
        -standard_normal_ppf(p), rel=RTOL
    )


def test_monotonic_increasing():
    ps = np.concatenate(
        [
            np.geomspace(5e-324, 1e-6, 50_000),
            np.linspace(1e-6, 1 - 1e-6, 200_001)[1:],
        ]
    )
    values = standard_normal_ppf(ps)
    assert np.all(np.diff(values) >= 0)


# --- Domain handling: same conventions as R qnorm / scipy --------------------


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, -math.inf),
        (1.0, math.inf),
        (-0.1, math.nan),
        (1.1, math.nan),
        (-math.inf, math.nan),
        (math.inf, math.nan),
        (math.nan, math.nan),
    ],
)
def test_edge_values_like_scipy(p, expected):
    result = standard_normal_ppf(p)
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(result, norm.ppf(p))


def test_edge_values_in_array_do_not_affect_valid_entries():
    p = np.array([0.0, 0.975, -1.0, 0.5, np.nan, 1.0, 0.025])
    result = standard_normal_ppf(p)
    np.testing.assert_allclose(result, norm.ppf(p), rtol=RTOL, equal_nan=True)


# --- Array support -----------------------------------------------------------


@pytest.mark.parametrize(
    "to_input",
    [list, tuple, lambda v: np.array(v, dtype=float)],
    ids=["list", "tuple", "ndarray"],
)
def test_array_like_input_returns_ndarray(to_input):
    values = [0.001, 0.025, 0.5, 0.8, 0.975, 1 - 1e-12]
    result = standard_normal_ppf(to_input(values))
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(values),)
    np.testing.assert_allclose(result, norm.ppf(values), rtol=RTOL)


def test_array_matches_scalar_calls():
    values = np.geomspace(1e-300, 0.5, 1000)
    values = np.concatenate([values, 1.0 - values])
    vectorised = standard_normal_ppf(values)
    scalar = np.array([standard_normal_ppf(float(v)) for v in values])
    np.testing.assert_array_equal(vectorised, scalar)


def test_multidimensional_shape_is_preserved():
    p = np.array([[0.025, 0.5, 0.975], [1e-10, 0.3, 1 - 1e-10]])
    result = standard_normal_ppf(p)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, norm.ppf(p), rtol=RTOL)


def test_integer_array_input():
    np.testing.assert_equal(standard_normal_ppf([0, 1]), [-np.inf, np.inf])


def test_empty_input():
    result = standard_normal_ppf([])
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


@pytest.mark.parametrize(
    "p", [0.975, np.float64(0.975), np.float32(0.975), np.array(0.975)]
)
def test_scalar_input_returns_python_float(p):
    result = standard_normal_ppf(p)
    assert type(result) is float
    # Compare in float64: scipy keeps float32 input in float32 precision.
    assert result == pytest.approx(float(norm.ppf(float(p))), rel=RTOL)


def test_input_array_is_not_modified():
    p = np.array([0.0, 0.1, 0.9, 1.0])
    original = p.copy()
    standard_normal_ppf(p)
    np.testing.assert_array_equal(p, original)
