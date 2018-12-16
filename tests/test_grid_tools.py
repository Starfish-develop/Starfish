import pytest

import Starfish.constants as C
from Starfish.grid_tools import *


class TestRawGridInterface:

    @pytest.fixture
    def rawgrid(self):
        params = {
            "temp": {5000, 5100, 5200},
            "logg": {3.0, 3.5, 4.0},
            "Z"   : {-0.5, 0.0}
        }
        yield RawGridInterface("PHOENIX",
                               list(params),
                               params.values(),
                               wl_range=(3000, 54000))

    @pytest.mark.skip("There is no implementation of this in the code, but could be useful. At the super-class level "
                      "it's tough to be able to check all params for sensible keys")
    def test_initialize(self):
        params = {
            "temp": {5000, 5100, 5200},
            "logg": {3.0, 3.5, 4.0},
            "Z"   : {-0.5, 0.0},
            "bunnies": ["furry", "happy"],
        }
        with pytest.raises(KeyError) as e:
            RawGridInterface("PHOENIX",
                             list(params),
                             params.values(),
                             wl_range=(3000, 54000))

    def test_check_params(self, rawgrid):
        rawgrid.check_params((5100,3.5, 0.0))

    def test_check_params_extra(self, rawgrid):
        with pytest.raises(ValueError) as e:
            rawgrid.check_params({"temp": 5100, "logg": 3.5, "bunny": 0.0})

    def test_check_params_out_of_bounds(self, rawgrid):
        with pytest.raises(ValueError) as e:
            rawgrid.check_params({"temp": 100, "logg": 3.5, "Z": 0.0})

    def test_implementation_error(self, rawgrid):
        with pytest.raises(NotImplementedError):
            rawgrid.load_flux({"temp": 100, "logg": 3.5, "Z": 0.0})


class TestPHOENIXGridInterface:

    @pytest.fixture
    def grid(self):
        yield PHOENIXGridInterface()

    def test_check_params_alpha(self, grid):
        grid.check_params({"temp": 5100, "logg": 3.5, "Z": 0.0, "alpha": 0.2})

    def test_load_flux(self, grid):
        fl, header = grid.load_flux({"temp": 5100, "logg": 3.5, "Z": 0.0, "alpha": 0.0})
        assert len(fl) == 1569128
        assert header['PHXTEFF'] == 5100
        assert header['PHXLOGG'] == 3.5
        assert header['PHXM_H'] == 0.0
        assert header['PHXALPHA'] == 0.0

    def test_load_alpha(self, grid):
        grid.load_flux({"temp": 6000, "logg": 3.5, "Z": 0.0})
        grid.load_flux({"temp": 6000, "logg": 3.5, "Z": 0.0, "alpha": 0.0})

    def test_load_flux_metadata(self, grid):
        spec = grid.load_flux({"temp": 6000, "logg": 3.5, "Z": 0.0, "alpha": 0.0})
        print(spec.metadata)

    def test_bad_base(self):
        # Set a different base location, should raise an error because on this machine there is not file.
        with pytest.raises(ValueError) as e:
            PHOENIXGridInterface(air=True, base="wrong_base/")

    def test_no_air(self):
        grid = PHOENIXGridInterface(air=False)
        spec = grid.load_flux({"temp": 5100, "logg": 3.5, "Z": 0.0, "alpha": 0.0})
        assert spec.metadata['air'] == False

    def test_no_norm(self, grid):
        spec = grid.load_flux({"temp": 5100, "logg": 3.5, "Z": 0.0, "alpha": 0.0}, norm=False)
        assert spec.metadata['norm'] == False
