import itertools

import numpy as np
import pytest

from Starfish.emulator._utils import flatten_parameters, deflatten_parameters


class TestEmulator:

    def test_creation(self, mock_emulator):
        assert mock_emulator._trained == False

    def test_call(self, mock_emulator):
        mu, cov = mock_emulator([6020, 4.21, -0.01])
        assert mu.shape == (6,)
        assert cov.shape == (6, 6)

    def test_load_flux(self, mock_emulator):
        flux, sigma = mock_emulator.load_flux([6020, 4.21, -0.01], full_cov=False)
        assert flux.shape == sigma.shape

    def test_load_flux_full_cov(self, mock_emulator):
        flux, sigma = mock_emulator.load_flux([6020, 4.21, -0.01], full_cov=True)
        assert sigma.shape == (len(flux), len(flux))

    def test_drawing_weights(self, mock_emulator):
        params = list(itertools.product(
            [6000, 6050, 6100],
            [4.0, 4.2],
            [-0.5]
        ))
        weights = mock_emulator.draw_many_weights(params)
        assert len(weights) == len(params)

    def test_warns_before_trained(self, mock_emulator):
        with pytest.warns(UserWarning):
            mock_emulator([6000, 4.2, 0.0])


    def test_train(self, mock_emulator):
        initial = flatten_parameters(mock_emulator.lambda_xi, mock_emulator.variances, mock_emulator.lengthscales)
        init_v11 = mock_emulator.v11_cho[0]
        assert mock_emulator._trained == False
        mock_emulator.train()
        final = flatten_parameters(mock_emulator.lambda_xi, mock_emulator.variances, mock_emulator.lengthscales)
        final_v11 = mock_emulator.v11_cho[1]
        assert not np.allclose(final_v11, init_v11)
        assert not np.allclose(final, initial)
        assert mock_emulator._trained == True

class TestUtils:

    def test_flatten_deflatten(self):
        lx = 1
        vs = np.random.randn(6) + 10
        ls = np.random.randn(6, 3) + 10
        lx_out, vs_out, ls_out = deflatten_parameters(flatten_parameters(lx, vs, ls), 6)
        assert lx_out == lx
        assert np.allclose(vs_out, vs)
        assert np.allclose(ls_out, ls)