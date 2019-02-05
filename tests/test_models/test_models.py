import pytest

from Starfish.models import SpectrumModel

class TestSpectrumModel:

    @pytest.fixture
    def mock_model(self, mock_data_spectrum, mock_trained_emulator):
        yield SpectrumModel(mock_trained_emulator, mock_data_spectrum)

    def test_transform(self, mock_model, mock_parameter):
        flux, cov = mock_model(mock_parameter)
        assert cov.shape == (len(flux), len(flux))
        assert flux.shape == mock_model.wave.shape

