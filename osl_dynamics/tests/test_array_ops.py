from osl_dynamics import array_ops
import numpy as np
import pytest
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
    values = np.array([[0.1, 0.2, 0.1, 0.3], [0.2, 0.1, 0.3, 0.1], [0.3, 0.2, 0.1, 0.1]])
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

def test_cov2std():
    random_signals = [np.random.default_rng().random(size=(5, 100)) for _ in range(7)]
    covs = np.array([np.cov(signal) for signal in random_signals])
    stds = np.array([np.std(signal, axis=1) for signal in random_signals])

    assert np.allclose(array_ops.cov2std(covs), stds)
