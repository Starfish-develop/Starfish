import numpy as np
import pytest

from Starfish.emulator.covariance import rbf_kernel, batch_kernel
@pytest.fixture
def mock_params():
    params = np.array((100, 1, 0.1), dtype=np.double) * \
        np.random.randn(200, 3) + np.tile((6000, 4, 0), (200, 1))
    return params


@pytest.fixture
def mock_kern_params():
    variances = np.ones(6)
    lengthscales = np.ones((6, 3))
    return variances, lengthscales


class TestKernel:

    def test_rbf_kernel_same(self, mock_params, mock_kern_params):
        variances, lengthscales = mock_kern_params
        cov = rbf_kernel(mock_params, mock_params,
                         variances[0], lengthscales[0])
        assert cov.shape == (len(mock_params), len(mock_params))

    def test_rbf_kernel_diff(self, mock_params, mock_kern_params):
        variances, lengthscales = mock_kern_params
        other_params = mock_params[10:50]
        cov = rbf_kernel(mock_params, other_params,
                         variances[0], lengthscales[0])
        assert cov.shape == (len(mock_params), len(other_params))


class TestBatchKernel:

    def test_batch_kernel_same(self, mock_params, mock_kern_params):
        variances, lengthscales = mock_kern_params
        cov = batch_kernel(mock_params, mock_params, variances, lengthscales)
        assert cov.shape == (len(variances) * len(mock_params),
                             len(variances) * len(mock_params))

    def test_batch_kernel_diff(self, mock_params, mock_kern_params):
        variances, lengthscales = mock_kern_params
        other_params = mock_params[10:50]
        cov = batch_kernel(mock_params, other_params, variances, lengthscales)
        assert cov.shape == (len(variances) * len(mock_params),
                             len(variances) * len(other_params))
