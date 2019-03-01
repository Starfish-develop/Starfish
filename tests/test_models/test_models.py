import pytest
import numpy as np

from Starfish.models import SpectrumModel

class TestSpectrumModel:

    def test_transform(self, mock_model):
        flux, cov = mock_model()
        assert cov.shape == (len(flux), len(flux))
        assert flux.shape == mock_model.wave.shape

    def test_warns_without_data(self, mock_emulator):
        with pytest.warns(UserWarning):
            model = SpectrumModel(mock_emulator, [6000, 4.0, 0])

    def test_get_set_parameters(self, mock_model):
        P0 = mock_model.get_parameter_vector()
        mock_model.set_parameter_vector(P0)
        P1 = mock_model.get_parameter_vector()
        assert np.allclose(P1, P0)

    def test_get_set_param_dict(self, mock_model):
        P0 = mock_model.get_parameter_dict()
        mock_model.set_parameter_dict(P0)
        P1 = mock_model.get_parameter_dict()
        assert P0 == P1

    def test_log_likelihood(self, mock_model):
        logl = mock_model.log_likelihood()
        assert np.isfinite(logl)

    def test_grad_log_likelihood(self, mock_model):
        with pytest.raises(NotImplementedError):
            mock_model.grad_log_likelihood()

    def test_save_load(self, mock_model, tmpdir):
        path = tmpdir.join('model.hdf5')
        mock_model.save(path)
        model = SpectrumModel.load(path)
        assert model == mock_model