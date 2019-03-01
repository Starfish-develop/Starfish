import numpy as np
import pytest
from scipy.linalg import block_diag

from Starfish.emulator._utils import inverse_block_diag
from Starfish.emulator import Emulator

class TestEmulator:

    def test_creation(self, mock_emulator):
        assert mock_emulator._trained == False

    def test_call(self, mock_emulator):
        mu, cov = mock_emulator([6020, 4.21, -0.01])
        assert mu.shape == (mock_emulator.ncomps,)
        assert cov.shape == (mock_emulator.ncomps, mock_emulator.ncomps)

    def test_call_multiple(self, mock_emulator):
        params = [
            [6020, 4.21, -0.01],
            [6104, 4.01, -0.23]
        ]
        n = mock_emulator.ncomps * len(params)
        mu, cov = mock_emulator(params)
        assert mu.shape == (n,)
        assert cov.shape == (n, n)

    def test_reinterpret_dims_fails(self, mock_emulator):
        params = [
            [6020, 4.21, -0.01],
            [6104, 4.01, -0.23],
            [6054, 4.15, -0.16]
        ]
        with pytest.raises(ValueError):
            mock_emulator(params, full_cov=True, reinterpret_batch=True)

    def test_reinterpret_dims(self, mock_emulator):
        params = [
            [6020, 4.21, -0.01],
            [6104, 4.01, -0.23],
            [6054, 4.15, -0.16]
        ]
        batch_mus, batch_vars = mock_emulator(params, full_cov=False, reinterpret_batch=True)
        lin_mus = np.empty((3, mock_emulator.ncomps))
        lin_vars = np.empty((3, mock_emulator.ncomps))
        for i, p in enumerate(params):
            lin_mus[i], lin_vars[i] = mock_emulator(p, full_cov=False, reinterpret_batch=True)
        assert np.allclose(batch_mus, lin_mus)
        assert np.allclose(batch_vars, lin_vars)

    def test_std(self, mock_emulator):
        mu, std = mock_emulator([6020, 4.21, -0.01], full_cov=False)
        assert mu.shape == (mock_emulator.ncomps,)
        assert std.shape == (mock_emulator.ncomps,)

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
        init_ll = mock_emulator.log_likelihood()
        init_v11 = mock_emulator.v11
        assert mock_emulator._trained == False
        mock_emulator.train(options={'maxiter': 10})
        final = mock_emulator.get_param_vector()
        final_ll = mock_emulator.log_likelihood()
        final_v11 = mock_emulator.v11
        assert not np.allclose(final_v11, init_v11)
        assert not np.allclose(final, initial)
        assert final_ll > init_ll

    def test_get_set_param_vector(self, mock_emulator):
        P0 = mock_emulator.get_param_vector()
        mock_emulator.set_param_vector(P0)
        P1 = mock_emulator.get_param_vector()
        assert np.allclose(P0, P1)

    def test_log_likelihood(self, mock_emulator):
        ll = mock_emulator.log_likelihood()
        assert np.isfinite(ll)

    def test_save_load(self, mock_emulator, tmpdir):
        init = mock_emulator.get_param_vector()
        filename = tmpdir.join('emu.hdf5')
        mock_emulator.save(filename)
        emulator = Emulator.load(filename)
        final = emulator.get_param_vector()
        assert np.allclose(init, final)
        assert emulator._trained == mock_emulator._trained

    def test_bulk_flux(self, mock_emulator):
        fluxes = mock_emulator.bulk_fluxes
        assert fluxes.shape == (mock_emulator.ncomps + 2, mock_emulator.eigenspectra.shape[-1])

class TestUtils:

    def test_inverse_block_diag(self):
        n = 4
        blocks = 10 * np.random.randn(n, 6, 6)
        block_matrix = block_diag(*blocks)
        out = inverse_block_diag(block_matrix, n)
        assert np.allclose(out, blocks)
