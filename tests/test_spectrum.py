import os

import numpy as np
import pytest

from Starfish import Spectrum
from Starfish.spectrum import Order


class TestOrder:
    def test_no_sigma(self, mock_data):
        wave, flux = mock_data
        order = Order(wave, flux)
        assert np.all(order._sigma == 0.0)
        assert np.all(order.mask)

    def test_no_mask(self, mock_data):
        wave, flux = mock_data
        sigma = np.random.randn(len(wave))
        order = Order(wave, flux, sigma)
        assert np.all(order.mask)

    def test_masking(self, mock_data):
        wave, flux = mock_data
        sigma = np.random.randn(len(wave))
        mask = sigma > 0
        order = Order(wave, flux, sigma, mask)
        assert np.allclose(order.wave, wave[mask])
        assert np.allclose(order.flux, flux[mask])
        assert np.allclose(order.sigma, sigma[mask])

    def test_len(self, mock_data):
        wave, flux = mock_data
        order = Order(wave, flux)
        assert len(order) == len(wave)


class TestSpectrum:
    def test_reshaping(self):
        waves = [np.linspace(1e4, 2e4, 100), np.linspace(2e4, 3e4, 100)]
        fluxes = [np.sin(waves[0]), np.cos(waves[1])]
        wave = np.hstack(waves)
        flux = np.hstack(fluxes)
        data = Spectrum(wave, flux, name="single")
        assert data.shape == (1, 200)
        reshaped = data.reshape((2, -1))
        assert reshaped.shape == ((2, 100))
        assert reshaped.name == "single"
        assert np.allclose(reshaped.waves, waves)
        assert np.allclose(reshaped.fluxes, fluxes)
        assert reshaped.shape == (2, 100)

        data.shape = (2, -1)

        assert np.allclose(reshaped.waves, data.waves)
        assert np.allclose(reshaped.fluxes, data.fluxes)

    def test_dunders(self, mock_spectrum):
        assert len(mock_spectrum) == 1
        assert isinstance(mock_spectrum[0], Order)
        mock_spectrum.name = "special"
        assert str(mock_spectrum).startswith("special")
        for i, order in enumerate(mock_spectrum):
            assert order == mock_spectrum[i]
        reshape_spec = mock_spectrum.reshape((2, -1))
        reshape_spec[0], reshape_spec[1] = reshape_spec[1], reshape_spec[0]

    def test_set_ragged_length(self, mock_spectrum):
        reshaped_spec = mock_spectrum.reshape((2, -1))
        with pytest.raises(ValueError):
            mock_spectrum[0] = reshaped_spec[0]

    def test_save_load(self, mock_spectrum, tmpdir):
        filename = os.path.join(tmpdir, "data.hdf5")
        mock_spectrum.save(filename)
        new_spectrum = Spectrum.load(filename)
        assert np.all(new_spectrum.waves == mock_spectrum.waves)
        assert np.all(new_spectrum.fluxes == mock_spectrum.fluxes)
        assert np.all(new_spectrum.sigmas == mock_spectrum.sigmas)
        assert np.all(new_spectrum.masks == mock_spectrum.masks)
