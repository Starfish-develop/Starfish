import pytest

import numpy as np

from Starfish.models.likelihoods import order_likelihood


class TestLikelihoods:
    def test_order_likelihood(self, mock_model):
        lnprob = order_likelihood(mock_model)
        assert np.isfinite(lnprob)
        assert lnprob.shape == ()

    def test_exact_order_likelihood(self, mock_model):
        lnprob = order_likelihood(mock_model)
        y, cov = mock_model()
        mock_model.data._fluxes = np.atleast_2d(y)
        lnprob_exact = order_likelihood(mock_model)
        assert lnprob_exact > lnprob
