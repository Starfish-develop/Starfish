import os

import pytest

from Starfish.grid_tools import RawGridInterface, PHOENIXGridInterface, PHOENIXGridInterfaceNoAlpha, \
    PHOENIXGridInterfaceNoAlphaNoFE

def test_phoenix_downloads(PHOENIXModels, AlphaPHOENIXModels):
    num_files = sum([len(files) for d, dd, files in os.walk(PHOENIXModels)])
    assert num_files == 29


class TestRawGridInterface:

    @pytest.fixture(scope='class')
    def rawgrid(self):
        params = {
            "temp": (6000, 6100, 6200),
            "logg": (4.0, 4.5, 5.0),
            "Z"   : (-0.5, 0.0)
        }
        yield RawGridInterface("PHOENIX",
                               list(params),
                               params.values(),
                               wl_range=(3000, 64000))

    @pytest.mark.skip("There is no implementation of this in the code, but could be useful. At the super-class level "
                      "it's tough to be able to check all params for sensible keys")
    def test_initialize(self):
        params = {
            "temp"   : (6000, 6100, 6200),
            "logg"   : (4.0, 4.5, 5.0),
            "Z"      : (-0.5, 0.0),
            "bunnies": ("furry", "happy"),
        }
        with pytest.raises(KeyError) as e:
            RawGridInterface("PHOENIX",
                             list(params),
                             params.values(),
                             wl_range=(3000, 54000))

    def test_check_params(self, rawgrid):
        rawgrid.check_params((6100, 4.5, 0.0))

    def test_check_params_extra(self, rawgrid):
        with pytest.raises(ValueError) as e:
            rawgrid.check_params((6100, 4.5, 0.0, 'John Cena'))

    def test_check_params_out_of_bounds(self, rawgrid):
        with pytest.raises(ValueError) as e:
            rawgrid.check_params((6100, 4.5, 0.5))

    def test_implementation_error(self, rawgrid):
        with pytest.raises(NotImplementedError):
            rawgrid.load_flux((6100, 4.5, 0.0))


class TestPHOENIXGridInterface:

    @pytest.fixture(scope='class')
    def grid(self, PHOENIXModels, AlphaPHOENIXModels):
        yield PHOENIXGridInterface(base=PHOENIXModels)

    def test_check_params_alpha(self, grid):
        assert grid.check_params((6100, 4.5, 0.0, -0.2))
        with pytest.raises(ValueError):
            assert grid.check_params((6100, 4.5, 0.0, 0.1))

    def test_load_flux(self, grid):
        fl, header = grid.load_flux((6100, 4.5, 0.0, 0.0))
        assert len(fl) == 1540041
        assert header['PHXTEFF'] == 6100
        assert header['PHXLOGG'] == 4.5
        assert header['PHXM_H'] == 0.0
        assert header['PHXALPHA'] == 0.0

    def test_load_alpha(self, grid):
        fl, header = grid.load_flux((6100, 4.5, 0.0, -0.2))
        assert header['PHXALPHA'] == -0.2

    def test_load_flux_metadata(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5, 0.0, 0.0))
        assert isinstance(hdr, dict)

    def test_bad_base(self):
        # Set a different base location, should raise an error because on this machine there is not file.
        with pytest.raises(ValueError) as e:
            PHOENIXGridInterface(base="wrong_base/")

    def test_no_air(self, PHOENIXModels):
        grid = PHOENIXGridInterface(air=False, base=PHOENIXModels)
        fl, hdr = grid.load_flux((6100, 4.5, 0.0, 0.0))
        assert hdr['air'] == False

    def test_no_norm(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5, 0.0, 0.0), norm=False)
        assert hdr['norm'] == False

    def test_no_header(self, grid):
        fl = grid.load_flux((6100, 4.5, 0.0, 0.0), header=False)
        assert len(fl) == 1540041


class TestPHOENIXGridInterfaceNoAlpha:

    @pytest.fixture(scope='class')
    def grid(self, PHOENIXModels):
        yield PHOENIXGridInterfaceNoAlpha(base=PHOENIXModels)

    def test_overload_params_failure(self, grid):
        with pytest.raises(ValueError):
            grid.load_flux((6100, 4.5, 0.0, 0.2))

    def test_load_flux(self, grid):
        fl, header = grid.load_flux((6100, 4.5, 0.0))
        assert len(fl) == 1540041
        assert header['PHXTEFF'] == 6100
        assert header['PHXLOGG'] == 4.5
        assert header['PHXM_H'] == 0.0
        assert header['PHXALPHA'] == 0.0

    def test_load_flux_metadata(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5, 0.0))
        assert isinstance(hdr, dict)

    def test_no_air(self, PHOENIXModels):
        grid = PHOENIXGridInterfaceNoAlpha(air=False, base=PHOENIXModels)
        fl, hdr = grid.load_flux((6100, 4.5, 0.0))
        assert hdr['air'] == False

    def test_no_norm(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5, 0.0), norm=False)
        assert hdr['norm'] == False

    def test_no_header(self, grid):
        fl = grid.load_flux((6100, 4.5, 0.0), header=False)
        assert len(fl) == 1540041


class TestPHOENIXGridInterfaceNoAlphaNoFE:

    @pytest.fixture(scope='class')
    def grid(self, PHOENIXModels):
        yield PHOENIXGridInterfaceNoAlphaNoFE(base=PHOENIXModels)

    def test_overload_params_failure(self, grid):
        with pytest.raises(ValueError):
            grid.load_flux((6100, 4.5, 0.0, 0.2))
        with pytest.raises(ValueError):
            grid.load_flux((6100, 4.5, 0.0))

    def test_load_flux(self, grid):
        fl, header = grid.load_flux((6100, 4.5))
        assert len(fl) == 1540041
        assert header['PHXTEFF'] == 6100
        assert header['PHXLOGG'] == 4.5
        assert header['PHXM_H'] == 0.0
        assert header['PHXALPHA'] == 0.0

    def test_load_flux_metadata(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5))
        assert isinstance(hdr, dict)

    def test_no_air(self, PHOENIXModels):
        grid = PHOENIXGridInterfaceNoAlphaNoFE(air=False, base=PHOENIXModels)
        fl, hdr = grid.load_flux((6100, 4.5))
        assert hdr['air'] == False

    def test_no_norm(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5), norm=False)
        assert hdr['norm'] == False

    def test_no_header(self, grid):
        fl = grid.load_flux((6100, 4.5), header=False)
        assert len(fl) == 1540041
