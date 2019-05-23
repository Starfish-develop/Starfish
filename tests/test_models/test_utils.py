from collections import deque

import numpy as np
import pytest

from Starfish.models import find_residual_peaks, optimize_residual_peaks

class TestUtils:

    @pytest.fixture
    def model(self, mock_model):
        fake_residual = np.random.randn(100, *mock_model.data.waves.shape)
        fake_residual[:, 10] = 1000
        mock_model.residuals = deque(fake_residual.tolist())
        yield mock_model

    def test_find_residuals_peaks(self, model):
        peaks = find_residual_peaks(model, num_residuals=50)
        assert model.data.waves[10] == peaks[0]
        assert len(peaks) < len(model.data.waves) - 2

    def test_optimize_residual_peaks(self, model):
        peaks = find_residual_peaks(model, num_residuals=50)
        params = optimize_residual_peaks(model, mus=peaks, num_residuals=50)
        assert len(params) == len(peaks)
        model.local = params
        assert model.params['local'] == params