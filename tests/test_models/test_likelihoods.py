import pytest

import numpy as np

from Starfish.models.likelihoods import mvn_likelihood, normal_likelihood

class TestLikelihoods:

    def test_normal_likelihood(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        flux = y + np.random.randn(len(y))
        lnprob, R = normal_likelihood(flux, y, var)
        assert np.isfinite(lnprob)
        assert lnprob.shape == ()

    def test_exact_normal_likelihoo(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        flux = y + np.random.randn(len(y))
        lnprob, R = normal_likelihood(y, y, var)
        assert lnprob > normal_likelihood(flux, y, var)[0]

    def test_mvn_likelihood(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        cov = np.diag(var)
        flux = y + np.random.randn(len(y))
        lnprob, R = mvn_likelihood(flux, y, cov)
        assert np.isfinite(lnprob)
        assert lnprob.shape == ()

    def test_exact_mvn_likelihood(self, mock_data_spectrum):
        y  = mock_data_spectrum.fluxes
        var = mock_data_spectrum.sigmas ** 2
        cov = np.diag(var)
        flux = y + np.random.randn(len(y))
        lnprob, R = mvn_likelihood(y, y, cov)
        assert lnprob > mvn_likelihood(flux, y, cov)[0]
