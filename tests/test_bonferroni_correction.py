import numpy as np
import pytest

from ab_agent.stats.multiple_testing_corrections import bonferroni_correction


def _as_list(values):
    return list(values)


def _as_array(values):
    return np.array(values, dtype=float)


INPUT_TYPES = pytest.mark.parametrize(
    "to_input", [_as_list, _as_array, tuple], ids=["list", "ndarray", "tuple"]
)


@INPUT_TYPES
def test_basic_mask(to_input):
    # m = 4, alpha = 0.05 -> threshold 0.0125
    p = [0.001, 0.02, 0.01, 0.5]
    result = bonferroni_correction(to_input(p), 0.05)
    np.testing.assert_array_equal(result, [True, False, True, False])


@INPUT_TYPES
def test_returns_bool_ndarray_of_same_length(to_input):
    p = [0.001, 0.2, 0.03]
    result = bonferroni_correction(to_input(p), 0.05)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert result.shape == (3,)


@INPUT_TYPES
def test_boundary_is_significant(to_input):
    # m = 2, alpha = 0.1 -> threshold exactly 0.05
    result = bonferroni_correction(to_input([0.05, 0.0500001]), 0.1)
    np.testing.assert_array_equal(result, [True, False])


@INPUT_TYPES
@pytest.mark.parametrize("p, expected", [(0.0, True), (0.04, True), (0.9, False)])
def test_single_element(to_input, p, expected):
    result = bonferroni_correction(to_input([p]), 0.05)
    np.testing.assert_array_equal(result, [expected])
    assert result.dtype == bool


@INPUT_TYPES
def test_empty_input(to_input):
    result = bonferroni_correction(to_input([]), 0.05)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert result.shape == (0,)


@INPUT_TYPES
def test_all_significant(to_input):
    result = bonferroni_correction(to_input([0.0, 0.001, 0.005]), 0.05)
    np.testing.assert_array_equal(result, [True, True, True])


@INPUT_TYPES
def test_none_significant(to_input):
    result = bonferroni_correction(to_input([0.02, 0.3, 1.0]), 0.05)
    np.testing.assert_array_equal(result, [False, False, False])


@INPUT_TYPES
def test_preserves_input_order(to_input):
    # m = 5, alpha = 0.05 -> threshold 0.01
    p = [0.9, 0.001, 0.5, 0.01, 0.011]
    result = bonferroni_correction(to_input(p), 0.05)
    np.testing.assert_array_equal(result, [False, True, False, True, False])


def test_list_and_array_agree():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.05, size=20)
    np.testing.assert_array_equal(
        bonferroni_correction(p, 0.05), bonferroni_correction(p.tolist(), 0.05)
    )
