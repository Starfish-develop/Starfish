import numpy as np
import pytest

from Starfish.emulator.covariance import rbf_kernel, batch_kernel

class TestCovariance:

    @pytest.fixture
    def mock_params(self):
        params = np.array((100, 1, 0.1), dtype=np.double) * np.random.randn(200, 3) + np.tile((6000, 4, 0), (200, 1))
        return params

    @pytest.fixture
    def mock_kern_params(self):
        variances = np.ones(6)
        lengthscales = np.ones((6, 3))
        return variances, lengthscales

    def test_rbf_kernel_same(self, mock_params, mock_kern_params):
        variances, lengthscales = mock_kern_params
        cov = rbf_kernel(mock_params, mock_params, variances[0], lengthscales[0])
        assert cov.shape == (mock_params.shape[0], mock_params.shape[0])

    def test_rbf_kernel_diff(self, mock_params, mock_kern_params):
        variances, lengthscales = mock_kern_params
        other_params = mock_params[10:50]
        cov = rbf_kernel(mock_params, other_params, variances[0], lengthscales[0])
        assert cov.shape == (mock_params.shape[0], other_params.shape[0])