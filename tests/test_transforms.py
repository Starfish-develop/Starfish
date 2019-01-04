import pytest

import numpy as np

from Starfish.transforms import Transform, Truncate, truncate, InstrumentalBroaden, \
    instrumental_broaden, RotationalBroaden, rotational_broaden, Resample, resample, \
    NullTransform, DopplerShift, doppler_shift
from Starfish.utils import calculate_dv, create_log_lam_grid



class TestTransform:

    def test_not_implemented(self, mock_data):
        t = Transform()
        with pytest.raises(NotImplementedError):
            t(*mock_data)
        with pytest.raises(NotImplementedError):
            t.transform(*mock_data)


class TestNullTransform:

    def test_null_transform(self, mock_data):
        t = NullTransform()
        wave, flux = t(*mock_data)
        np.testing.assert_allclose(wave, mock_data[0])
        np.testing.assert_allclose(flux, mock_data[1])

class TestTruncate:

    def test_no_truncation(self, mock_data):
        t = Truncate()
        wave, flux = t(*mock_data)
        np.testing.assert_allclose(wave, mock_data[0])
        np.testing.assert_allclose(flux, mock_data[1])

    @pytest.mark.parametrize('wl_range, expected', [
        [(0, np.inf), (1e4, 5e4)],
        [(1e4, 2e4), (1e4, 2e4)],
        [(2e4, 6e5), (2e4, 5e4)],
    ])
    def test_truncation_no_buffer(self, wl_range, expected, mock_data):
        t = Truncate(wl_range=wl_range, buffer=0)
        wave, flux = t(*mock_data)
        assert wave.shape == flux.shape
        assert wave[0] >= expected[0]
        assert wave[-1] <= expected[-1]

    @pytest.mark.parametrize('wl_range, expected', [
        [(0, np.inf), (1e4, 5e4)],
        [(1e4, 2e4), (1e4, 2e4)],
        [(2e4, 6e5), (2e4, 5e4)],
    ])
    def test_bare_method(self, wl_range, expected, mock_data):
        t = Truncate(wl_range=wl_range, buffer=0)
        w1, f1 = t(*mock_data)
        w2, f2 = truncate(*mock_data, wl_range, 0)
        np.testing.assert_allclose(w1, w2)
        np.testing.assert_allclose(f1, f2)


class TestInstrumentalBroaden:

    @pytest.mark.parametrize('fwhm', [
        -20,
        -1.00,
        -np.finfo(np.float64).tiny
    ])
    def test_bad_fwhm(self, fwhm):
        with pytest.raises(ValueError):
            InstrumentalBroaden(fwhm)

    def test_0_fwhm(self, mock_data):
        t = InstrumentalBroaden(0)
        wave, flux = t(*mock_data)
        np.testing.assert_allclose(wave, mock_data[0])
        np.testing.assert_allclose(flux, mock_data[1])

    def test_inst_broadening_inst(self, mock_data, mock_instrument):
        t = InstrumentalBroaden(mock_instrument)
        assert t.inst == mock_instrument.FWHM
        wave, flux = t(*mock_data)
        assert wave.shape == flux.shape
        assert wave.shape == mock_data[0].shape
        assert flux.shape == mock_data[1].shape

    def test_inst_broadening_fwhm(self, mock_data):
        t = InstrumentalBroaden(400)
        wave, flux = t(*mock_data)
        assert wave.shape == flux.shape

    def test_helper_func(self, mock_data, mock_instrument):
        t = InstrumentalBroaden(mock_instrument)
        wave1, flux1 = t(*mock_data)
        wave2, flux2 = instrumental_broaden(*mock_data, mock_instrument)
        np.testing.assert_allclose(wave1, wave2)
        np.testing.assert_allclose(flux1, flux2)

class TestRotationalBroaden:

    @pytest.mark.parametrize('vsini', [
        -20,
        -1.00,
        -np.finfo(np.float64).tiny,
        0
    ])
    def test_bad_fwhm(self, vsini):
        with pytest.raises(ValueError):
            RotationalBroaden(vsini)

    def test_rot_broadening_inst(self, mock_data):
        t = RotationalBroaden(84)
        assert t.vsini == 84
        wave, flux = t(*mock_data)
        assert wave.shape == flux.shape
        assert wave.shape == mock_data[0].shape
        assert flux.shape == mock_data[1].shape

    def test_helper_func(self, mock_data, mock_instrument):
        t = RotationalBroaden(84)
        wave1, flux1 = t(*mock_data)
        wave2, flux2 = rotational_broaden(*mock_data, 84)
        np.testing.assert_allclose(wave1, wave2)
        np.testing.assert_allclose(flux1, flux2)

class TestResample:

    @pytest.mark.parametrize('wave', [
        np.linspace(-1, -0.5),
        np.linspace(0, 1e4)
    ])
    def test_bad_waves(self, wave):
        with pytest.raises(ValueError):
            Resample(wave)

    def test_resample(self, mock_data):
        dv = calculate_dv(mock_data[0])
        new_wave = create_log_lam_grid(dv, mock_data[0].min(), mock_data[0].max())['wl']
        t = Resample(new_wave)
        wave, flux = t(*mock_data)
        assert wave.shape == flux.shape
        np.testing.assert_allclose(wave, new_wave)

    def test_helper_func(self, mock_data):
        dv = calculate_dv(mock_data[0])
        new_wave = create_log_lam_grid(dv, mock_data[0].min(), mock_data[0].max())['wl']
        wave, flux = resample(*mock_data, new_wave)
        assert wave.shape == flux.shape
        np.testing.assert_allclose(wave, new_wave)


class TestDopplerShift:

    def test_no_change(self, mock_data):
        t = DopplerShift(0)
        wave, flux = t(*mock_data)
        np.testing.assert_allclose(wave, mock_data[0])
        np.testing.assert_allclose(flux, mock_data[1])

    def test_blueshift(self, mock_data):
        t = DopplerShift(-1e3)
        wave, flux = t(*mock_data)
        assert np.all(wave < mock_data[0])
        np.testing.assert_allclose(flux, mock_data[1])

    def test_redshift(self, mock_data):
        t = DopplerShift(1e3)
        wave, flux = t(*mock_data)
        assert np.all(wave > mock_data[0])
        np.testing.assert_allclose(flux, mock_data[1])

    def test_helper_func(self, mock_data):
        t = DopplerShift(1e3)
        wave1, flux1 = doppler_shift(*mock_data, 1e3)
        wave2, flux2 = t(*mock_data)
        np.testing.assert_allclose(wave1, wave2)
        np.testing.assert_allclose(flux1, flux2)