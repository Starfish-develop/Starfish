import numpy as np

from Starfish.models.kernels import k_global_matrix

def test_global_matrix():
    wave = np.linspace(1e4, 2e4, 1000)
    a = 10
    l=1
    cov = k_global_matrix(wave, a, l)
    assert np.all(np.diag(cov) == a ** 2)
    assert cov.shape == (1000, 1000)
    assert np.all(np.diag(cov, k=-1) < a ** 2)
    assert np.all(np.diag(cov, k=1) < a ** 2)
    assert cov.min() == 0
    assert np.all(cov >= 0)