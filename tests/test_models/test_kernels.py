import numpy as np

from Starfish.models.kernels import global_covariance_matrix, local_covariance_matrix


def test_global_matrix():
    wave = np.linspace(1e4, 2e4, 1000)
    a = 100
    l = 1
    cov = global_covariance_matrix(wave, a, l)
    assert np.all(cov.diagonal() == a)
    assert cov.shape == (1000, 1000)
    assert np.all(np.diag(cov, k=-1) < a)
    assert np.all(np.diag(cov, k=1) < a)
    assert cov.min() == 0
    assert np.all(cov >= 0)
    assert cov.max() == a
    assert np.all(np.linalg.eigvals(cov) >= 0)
    assert np.allclose(cov, cov.T)


def test_local_matrix():
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
    ## Need to figure this out
    # assert np.all(np.linalg.eigvals(cov) >= 0)
