import pytest

from Starfish.grid_tools import *
import numpy as np
from Starfish.spectrum import create_log_lam_grid
import Starfish.constants as C

class TestRawGridInterface:
    def setup_class(self):
        self.rawgrid = RawGridInterface("PHOENIX", {"temp":{5000, 5100, 5200}, "logg":{3.0, 3.5, 4.0}, "Z":{-0.5, 0.0}})

    def test_initialize(self):
        with pytest.raises(KeyError) as e:
            RawGridInterface("PHOENIX", {"temp":{5000, 5100, 5200}, "logg":{3.0, 3.5, 4.0}, "Z":{-0.5, 0.0},
                                     'bunnies': {"furry", "happy"}})
        print(e.value)

    def test_check_params(self):
        self.rawgrid.check_params({"temp":5100, "logg":3.5, "Z":0.0})

    def test_check_params_extra(self):
        with pytest.raises(C.GridError) as e:
            self.rawgrid.check_params({"temp":5100, "logg":3.5, "bunny":0.0})
        print(e.value)

    def test_check_params_outbouts(self):
        with pytest.raises(C.GridError) as e:
            self.rawgrid.check_params({"temp":100, "logg":3.5, "Z":0.0})
        print(e.value)

class TestPHOENIXGridInterface:
    def setup_class(self):
        self.rawgrid = PHOENIXGridInterface()

    def test_check_params_alpha(self):
        self.rawgrid.check_params({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.2})

    def test_load_file(self):
        self.rawgrid.load_file({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})

        #not expected to be on leo, yet within bounds
        with pytest.raises(C.GridError) as e:
            self.rawgrid.load_file({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.2})
        print(e.value)

    def test_load_flux(self):
        fl, header = self.rawgrid.load_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})
        #print(fl, header)

    def test_load_alpha(self):
        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0})

        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})

    def test_load_file_metadata(self):
        spec = self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})
        print(spec.metadata)

    def test_bad_base(self):
        #Set a different base location, should raise an error because on this machine there is not file.
        with pytest.raises(C.GridError) as e:
            PHOENIXGridInterface(air=True, norm=True, base="wrong_base/")
        print(e.value)

    def test_no_air(self):
        grid = PHOENIXGridInterface(air=False)
        spec = grid.load_file({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})
        assert spec.metadata['air'] == False

    def test_no_norm(self):
        grid = PHOENIXGridInterface(norm=False)
        spec = grid.load_file({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})
        assert spec.metadata['norm'] == False
