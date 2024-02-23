from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from osl_dynamics import array_ops


def test_get_one_hot_simple():
    values = np.arange(5)
    assert (array_ops.get_one_hot(values) == np.identity(5)).all()


def test_get_one_hot_gap():
    values = np.array([0, 3])
    result = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    assert (array_ops.get_one_hot(values) == result).all()


def test_get_one_hot_extra():
    values = np.array([0, 1, 2])
    result = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    assert (array_ops.get_one_hot(values, n_states=4) == result).all()


def test_get_one_hot_fail():
    values = np.arange(5)
    # Arguably this should throw a better error.
    with pytest.raises(IndexError):
        array_ops.get_one_hot(values, n_states=3)


def test_get_one_hot_2d():
    values = np.array(
        [[0.1, 0.2, 0.1, 0.3], [0.2, 0.1, 0.3, 0.1], [0.3, 0.2, 0.1, 0.1]]
    )
    result = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
    assert (array_ops.get_one_hot(values) == result).all()


def test_cov2corr():
    random_signals = [np.random.default_rng().random(size=(5, 100)) for _ in range(7)]
    covs = np.array([np.cov(signal) for signal in random_signals])
    corrs = np.array([np.corrcoef(signal) for signal in random_signals])

    assert np.allclose(array_ops.cov2corr(covs), corrs)


def test_cov2corr_1d_fail():
    with pytest.raises(ValueError):
        array_ops.cov2corr(np.arange(10))


def test_cov2std_1d_fail():
    with pytest.raises(ValueError):
        array_ops.cov2std(np.array([1, 2, 3]))


def test_cov2std_success():
    n_batches = 3
    n_signals = 5
    n_timepoints = 100

    random_signals = [
        np.random.default_rng().random(size=(n_timepoints, n_signals))
        for _ in range(n_batches)
    ]
    covs = np.array([np.cov(signal, rowvar=False) for signal in random_signals])
    stds = np.array([np.std(signal, axis=0, ddof=1) for signal in random_signals])

    calc_stds = array_ops.cov2std(covs)

    assert calc_stds.shape == stds.shape
    np.testing.assert_allclose(calc_stds, stds)


@pytest.mark.parametrize(
    argnames="input_array,output_shape,expectation",
    argvalues=[
        (np.zeros((3, 4, 5)), (3, 4, 5), does_not_raise()),
        (np.zeros((4, 5)), (1, 4, 5), does_not_raise()),
        (np.zeros((3, 4, 5, 6)), None, pytest.raises(ValueError)),
    ],
    ids=["correct_dimensionality", "expand_dims", "incorrect_dimensionality"],
)
def test_validate_parameterized(input_array, output_shape, expectation):
    correct_dimensionality = 3
    allow_dimensions = [2, 3]
    error_message = "Array dimensionality is incorrect."

    with expectation:
        result = array_ops.validate(
            input_array, correct_dimensionality, allow_dimensions, error_message
        )
        assert result.shape == output_shape
