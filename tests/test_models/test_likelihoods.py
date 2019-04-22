import pytest

import numpy as np

from Starfish.models._likelihoods import mvn_likelihood, normal_likelihood

class TestLikelihoods:

    def test_normal_likelihood(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        flux = y + np.random.randn(len(y))
        lnprob = normal_likelihood(flux, y, var)
        assert np.isfinite(lnprob)
        assert lnprob.shape == ()
        assert lnprob < 0

    def test_exact_normal_likelihoo(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        flux = y + np.random.randn(len(y))
        lnprob = normal_likelihood(y, y, var)
        assert lnprob > normal_likelihood(flux, y, var)

    def test_mvn_likelihood(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        cov = np.diag(var)
        flux = y + np.random.randn(len(y))
        lnprob = mvn_likelihood(flux, y, cov)
        assert np.isfinite(lnprob)
        assert lnprob.shape == ()
        assert lnprob < 0

    def test_exact_mvn_likelihood(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        cov = np.diag(var)
        flux = y + np.random.randn(len(y))
        lnprob = mvn_likelihood(y, y, cov)
        assert lnprob > mvn_likelihood(flux, y, cov)
