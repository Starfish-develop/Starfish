import numpy as np
import pytest

from Starfish.constants import hc_k
from Starfish.models.kernels import global_covariance_matrix, local_covariance_matrix


class TestKernels:
    def test_global_matrix(self):
        wave = np.linspace(1e4, 2e4, 1000)
        amp = 100
        lengthscale = 1
        cov = global_covariance_matrix(wave, amp, lengthscale)
        assert np.allclose(cov.diagonal(), amp)
        assert cov.shape == (1000, 1000)
        assert np.all(np.diag(cov, k=-1) < amp)
        assert np.all(np.diag(cov, k=1) < amp)
        assert cov.min() == 0
        assert np.all(cov >= 0)
        assert np.allclose(
            cov.max(), amp
        )
        assert np.all(np.linalg.eigvals(cov) >= 0)
        assert np.allclose(cov, cov.T)

    def test_local_matrix(self):
        wave = np.linspace(1e4, 2e4, 1000)
        amp = 100
        mu = 1.5e4
        sigma = 1e3
        cov = local_covariance_matrix(wave, amp, mu, sigma)
        assert cov.shape == (1000, 1000)
        assert np.all(np.diag(cov, k=-1) < amp)
        assert np.all(np.diag(cov, k=1) < amp)
        assert cov.min() == 0
        assert np.all(cov >= 0)
        assert cov.max() <= amp
        assert np.allclose(cov, cov.T)
        # Need to figure this out
        # assert np.all(np.linalg.eigvals(cov) >= 0)


class TestKernelBenchmarks:
    @pytest.mark.parametrize("size", [100, 500, 1000, 2000, 5000, 10000])
    def test_local_matrix(self, benchmark, size):
        wave = np.linspace(1e4, 2e4, size)
        amp = 100
        mu = 1.5e4
        sigma = 1e3
        cov = benchmark(local_covariance_matrix, wave, amp, mu, sigma)
        assert cov.shape == (size, size)

    @pytest.mark.parametrize("size", [100, 500, 1000, 2000, 5000, 10000])
    def test_global_matrix(self, benchmark, size):
        wave = np.linspace(1e4, 2e4, size)
        amp = 100
        lengthscale = 1
        cov = benchmark(global_covariance_matrix, wave, amp, lengthscale)
        assert cov.shape == (size, size)
