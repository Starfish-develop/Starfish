import pytest

import numpy as np

from Starfish.transforms import Transform, Truncate, truncate, Broaden, broaden



class TestTransform:

    def test_not_implemented(self, mock_data):
        t = Transform()
        with pytest.raises(NotImplementedError):
            t(*mock_data)
        with pytest.raises(NotImplementedError):
            t._transform(*mock_data)

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


@pytest.mark.skip('need better understanding of math')
class TestBroaden:

    def test_no_broadening(self, mock_data):
        wave, flux = broaden(*mock_data, None, None, None)
        np.testing.assert_allclose(wave, mock_data[0])
        np.testing.assert_allclose(flux, mock_data[1])

    def test_inst_broadening_inst(self, mock_data, mock_instrument):
        t = Broaden(mock_instrument, None, None)
        assert t.inst == mock_instrument.FWHM
        wave, flux = t(*mock_data)
        assert wave.shape == flux.shape
        # assert wave.shape == mock_data[0].shape
        # assert flux.shape == mock_data[1].shape