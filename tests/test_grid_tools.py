from grid_tools import *

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
        print("Creating RawGridInterface")
        self.rawgrid = RawGridInterface("PHOENIX", {"temp":{5000, 5100, 5200}, "logg":{3.0, 3.5, 4.0}, "Z":{-0.5, 0.0}})

    def test_check_params(self):
        print("Parameter checking")
        self.rawgrid.check_params({"temp":5100, "logg":3.5, "Z":0.0})

    def test_check_params_extra(self):
        print("Checking to see if GridError properly raised")
        with pytest.raises(GridError) as e:
            self.rawgrid.check_params({"temp":5100, "logg":3.5, "bunny":0.0})
        print(e.value)

    def test_check_params_outside(self):
        print("Checking to see if GridError properly raised")
        with pytest.raises(GridError) as e:
            self.rawgrid.check_params({"temp":100, "logg":3.5, "Z":0.0})
        print(e.value)

class TestPHOENIXGridInterface(TestRawGridInterface):
    def setup_class(self):
        print("Creating PHOENIXGridInterface")
        self.rawgrid = PHOENIXGridInterface(air=True, norm=True)

    def test_check_params_alpha(self):
        print("Checking alpha")
        self.rawgrid.check_params({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.2})

    def test_load_alpha(self):
        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0})
        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})
        with pytest.raises(GridError) as e:
            self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.2})
        print(e.value)

    def test_load_file(self):
        print("Testing loading of file")
        spec = self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})
        print(spec.metadata)

class TestHDF5Creator:
    def setup_class(self):
        self.wldict = create_log_lam_grid(min_V=2.)
        self.rawgrid = PHOENIXGridInterface(air=True, norm=True)
        self.HDF5Creator = HDF5GridCreator(self.rawgrid, ranges={"temp":(5000, 6000), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                                            "alpha":(0.0, 0.0)}, filename="test.hdf5", wldict=self.wldict, chunksize=20)

    def test_point_limiter(self):
        print(self.HDF5Creator.points)

    def test_file_not_found(self):
        pass

    def test_process_grid(self):
        self.HDF5Creator.process_grid()
