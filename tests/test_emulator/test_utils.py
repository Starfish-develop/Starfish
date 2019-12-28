import numpy as np
import pytest
import scipy.stats as st
from sklearn.decomposition import PCA

from Starfish.emulator._utils import (
    get_w_hat,
    get_phi_squared,
    get_altered_prior_factors,
    Gamma,
)


class TestEmulatorUtils:
    @pytest.fixture
    def grid_setup(self, mock_hdf5_interface):
        fluxes = np.array(list(mock_hdf5_interface.fluxes))
        # Normalize to an average of 1 to remove uninteresting correlation
        fluxes /= fluxes.mean(1, keepdims=True)
        # Center and whiten
        flux_mean = fluxes.mean(0)
        fluxes -= flux_mean
        flux_std = fluxes.std(0)
        fluxes /= flux_std

        # Perform PCA using sklearn
        default_pca_kwargs = dict(n_components=0.99, svd_solver="full")
        pca = PCA(**default_pca_kwargs)
        weights = pca.fit_transform(fluxes)
        eigenspectra = pca.components_
        yield eigenspectra, fluxes

    def test_altered_lambda_xi(self, grid_setup):
        a_p, b_p = get_altered_prior_factors(*grid_setup)
        assert np.isfinite(a_p)
        assert np.isfinite(b_p)

    def test_w_hat(self, grid_setup):
        eigs, fluxes = grid_setup
        w_hat = get_w_hat(eigs, fluxes)
        assert len(w_hat) == len(fluxes) * len(eigs)
        assert np.all(np.isfinite(w_hat))

    def test_phi_squared(self, grid_setup):
        eigs, fluxes = grid_setup
        M = len(fluxes)
        m = len(eigs)
        phi2 = get_phi_squared(eigs, M)
        assert phi2.shape == (M * m, M * m)
        assert np.all(np.isfinite(phi2))

    @pytest.mark.parametrize("params", [(1, 0.001), (2, 0.075)])
    def test_gamma_dist(self, params):
        a, b = params
        mine = Gamma(a, b)
        theirs = st.gamma(a, scale=1 / b)
        x = np.linspace(1e-6, 1e4)
        assert np.allclose(mine.logpdf(x), theirs.logpdf(x))
