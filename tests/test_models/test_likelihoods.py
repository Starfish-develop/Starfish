import pytest
import numpy as np

from Starfish.models import SpectrumLikelihood, SpectrumParameter


class TestSpectrumLikelihood:

    def test_likelihood(self, mock_model, mock_parameter):
        like = SpectrumLikelihood(mock_model)
        lnp = like.log_probability(mock_parameter)
        assert lnp.shape == ()

    @pytest.mark.parametrize('param', [
        SpectrumParameter([6000, 4.2, 0.0], vz=0, vsini=-1),
        SpectrumParameter([6000, 4.2, 0.0], vz=0, Av=-3),
        SpectrumParameter([3000, 4.2, 0.0]),
        SpectrumParameter([-1000, 4.2, 0.0]),
        SpectrumParameter([3000, 4.2, 0.0]),
    ])
    def test_bad_parameters(self, mock_model, param):
        like = SpectrumLikelihood(mock_model)
        lnp = like.log_probability(param)
        assert lnp == -np.inf
