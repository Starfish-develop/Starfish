class TestSpectrumModel:

    def test_transform(self, mock_model, mock_parameter):
        flux, cov = mock_model(mock_parameter)
        assert cov.shape == (len(flux), len(flux))
        assert flux.shape == mock_model.wave.shape
