import pytest

from Starfish.grid_tools.instruments import *

class TestInstrumentBase:

    @pytest.fixture
    def mock_instrument(self):
        yield Instrument('Test Instrument', FWHM=45.0, wl_range=(300, 1e4))

    @pytest.mark.parametrize('attr', [
        'name',
        'FWHM',
        'oversampling',
        'wl_range'
    ])
    def test_attributes(self, attr, mock_instrument):
        assert hasattr(mock_instrument, attr)

    def test_string(self, mock_instrument):
        expected = "Instrument Name: Test Instrument, FWHM: 45.0, oversampling: 4, wl_range: (300, 10000.0)"
        assert str(mock_instrument) == expected


class TestSpecificInstruments:

    base_instruments = [IGRINS]
    instruments = [TRES, Reticon, KPNO, SPEX, SPEX_SXD , IGRINS_H, IGRINS_K, ESPaDOnS, DCT_DeVeny, WIYN_Hydra]

    @pytest.mark.parametrize('instrument', instruments)
    def test_can_create(self, instrument):
        assert instrument()

    @pytest.mark.parametrize('instrument', base_instruments)
    def test_base_cant_create(self, instrument):
        with pytest.raises(TypeError):
            assert instrument()