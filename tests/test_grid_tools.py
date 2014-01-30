import pytest
from StellarSpectra.grid_tools import *


def test_create_log_lam_grid():
    print(create_log_lam_grid(min_WL=(0.08,5000)))
    print(create_log_lam_grid(min_V=2))
    print(create_log_lam_grid(min_V=8))
    with pytest.raises(ValueError) as e:
        create_log_lam_grid()
    print(e.value)

def pytest_funcargs__valid_rawgrid_interface(request):
    return RawGridInterface("PHOENIX", {"temp":{5000, 5100, 5200}, "logg":{3.0, 3.5, 4.0}, "Z":{-0.5, 0.0}})

class TestRawGridInterface:
    def setup_class(self):
        print("Creating RawGridInterface\n")
        self.rawgrid = RawGridInterface("PHOENIX", {"temp":{5000, 5100, 5200}, "logg":{3.0, 3.5, 4.0}, "Z":{-0.5, 0.0}})

    def test_check_params(self):
        print("Checking default parameters in grid")
        self.rawgrid.check_params({"temp":5100, "logg":3.5, "Z":0.0})

    def test_check_params_extra(self):
        print("Checking to see if GridError properly raised with extra parameter\n")
        with pytest.raises(GridError) as e:
            self.rawgrid.check_params({"temp":5100, "logg":3.5, "bunny":0.0})
        print(e.value)

    def test_check_params_outside(self):
        print("Checking to see if GridError properly raised with out of bounds variable\n")
        with pytest.raises(GridError) as e:
            self.rawgrid.check_params({"temp":100, "logg":3.5, "Z":0.0})
        print(e.value)

class TestPHOENIXGridInterface(TestRawGridInterface):
    def setup_class(self):
        print("Creating PHOENIXGridInterface\n")
        self.rawgrid = PHOENIXGridInterface(air=True, norm=True)

    def test_check_params_alpha(self):
        print("using alpha in check_params\n")
        self.rawgrid.check_params({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.2})

    def test_load_alpha(self):
        print("loading files with and without alpha specified\n")
        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0})
        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})
        print("raises GridError if alpha is outside grid bounds\n")
        with pytest.raises(GridError) as e:
            self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.2})
        print(e.value)

    def test_load_file_metadata(self):
        print("Testing loading of file with metadata\n")
        spec = self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})
        print(spec.metadata, "/n")

    def test_base_location(self):
        print("Set a different base location, should raise an error because on this machine there is not file.")
        with pytest.raises(GridError) as e:
            self.newgrid = PHOENIXGridInterface(air=True, norm=True, base="wrong_base/")
            file = self.newgrid.load_file({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.2})
        print(e.value)


class TestHDF5Creator:
    def setup_class(self):
        self.wldict = create_log_lam_grid(min_V=0.2)
        self.rawgrid = PHOENIXGridInterface(air=True, norm=True)
        self.HDF5Creator = HDF5GridCreator(self.rawgrid, ranges={"temp":(5000, 6000), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                                            "alpha":(0.0, 0.0)}, filename="test.hdf5", wldict=self.wldict, chunksize=20)

    def test_point_limiter(self):
        print("HDF5Creator limiting points to ranges\n")
        print(self.HDF5Creator.points)

    def test_alpha(self):
        '''Test to make sure that alpha is handled properly, whether specified or not.'''
        self.alphaHDF5Creator = HDF5GridCreator(self.rawgrid, ranges={"temp":(5000, 6000), "logg":(3.5, 4.5), "Z":(0.0,0.0)},
                                           filename="test.hdf5", wldict=self.wldict, chunksize=20)
        pass

    def test_process_flux(self):
        self.HDF5Creator.process_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})
        print("process_flux requires all 4 parameters, including alpha\n")
        with pytest.raises(AssertionError) as e:
            self.HDF5Creator.process_flux({"temp":5100, "logg":3.5, "Z":0.0})
        print(e.value)

    def test_process_grid(self):
        print("Creating test grid \n")
        self.HDF5Creator.process_grid()

class TestHDF5Interface:
    def setup_class(self):
        self.interface = HDF5Interface("test.hdf5")

    def test_wl(self):
        print(self.interface.wl)
        print(self.interface.wl_header)
        #print(self.interface.flux_name_dict)

    def test_bounds(self):
        print(self.interface.bounds)

