import numpy as np
import pytest


class TestEmulator:

    def test_creation(self, mock_emulator):
        assert mock_emulator._trained == False

    def test_call(self, mock_emulator):
        mu, cov = mock_emulator([6020, 4.21, -0.01])
        assert mu.shape == (6,)
        assert cov.shape == (6, 6)

    def test_call_multiple(self, mock_emulator):
        params = [
            [6020, 4.21, -0.01],
            [6104, 4.01, -0.23]
        ]
        mu, cov = mock_emulator(params)
        assert mu.shape == (12,)
        assert cov.shape == (12, 12)

    def test_std(self, mock_emulator):
        mu, std = mock_emulator([6020, 4.21, -0.01], full_cov=False)
        assert mu.shape == (6,)
        assert std.shape == (6,)

    def test_load_flux(self, mock_emulator):
        flux = mock_emulator.load_flux([6020, 4.21, -0.01])
        assert np.all(np.isfinite(flux))

    def test_load_many_fluxes(self, mock_emulator):
        params = [
            [6020, 4.21, -0.01],
            [6104, 4.01, -0.23]
        ]
        flux = mock_emulator.load_flux(params)
        assert len(flux) == len(params)
        assert np.all(np.isfinite(flux))

    def test_warns_before_trained(self, mock_emulator):
        with pytest.warns(UserWarning):
            mock_emulator([6000, 4.2, 0.0])

    def test_train(self, mock_emulator):
        initial = mock_emulator.get_param_vector()
        init_v11 = mock_emulator.v11_cho[0]
        assert mock_emulator._trained == False
        mock_emulator.train(options={'maxiter': 10})
        final = mock_emulator.get_param_vector()
        final_v11 = mock_emulator.v11_cho[1]
        assert not np.allclose(final_v11, init_v11)
        assert not np.allclose(final, initial)
        assert mock_emulator._trained == True

    def test_get_set_param_vector(self, mock_emulator):
        P = mock_emulator.get_param_vector()
        emu = mock_emulator.set_param_vector(P)
        assert emu.lambda_xi == mock_emulator.lambda_xi
        assert np.allclose(emu.variances, mock_emulator.variances)
        assert np.allclose(emu.lengthscales, mock_emulator.lengthscales)

    def test_log_likelihood(self, mock_emulator):
        ll = mock_emulator.log_likelihood()
        assert np.isfinite(ll)
