import pytest
from StellarSpectra.grid_tools import *



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
        #TODO: test that air wavelengths are properly set?

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
        import StellarSpectra
        self.wldict = StellarSpectra.model.create_log_lam_grid(min_V=0.2)
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
        self.interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")

    def test_wl(self):
        print(self.interface.wl)
        print(self.interface.wl_header)
        #print(self.interface.flux_name_dict)

    def test_bounds(self):
        print(self.interface.bounds)

    def test_load_file(self):
        self.interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        pass


class TestIndexInterpolator:
    def setup_class(self):
        self.interpolator = IndexInterpolator([6000,6100,6200,6300,6400])

    def test_interpolate(self):
        ans = self.interpolator(6010)
        print(ans)
        assert ans == (0.1, 6000, 6100)

    def test_interpolate_bounds(self):
        with pytest.raises(InterpolationError) as e:
            self.interpolator(5990)
        print(e.value)


class TestInterpolator:
    def setup_class(self):
        hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        #libraries/PHOENIX_submaster.hd5 should have the following bounds
        #{"temp":(6000, 7000), "logg":(3.5,5.5), "Z":(-1.0,0.0), "alpha":(-0.2,0.0)}
        self.interpolator = Interpolator(hdf5interface)
        pass

    def test_parameters(self):
        print(self.interpolator.parameters)

    def test_trilinear(self):
        pass

    def test_interpolate_bounds(self):
        pass

    def test_quadlinear(self):
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1, "alpha":-0.1}
        new_flux = self.interpolator(parameters)
        #Use IPython and  %timeit -n1 -r1 mytest.interpolator({"temp":6010, "logg":5.1, "Z":-0.1, "alpha":-0.1})
        #all uncached performance is 3.89 seconds
        #1/2 uncached performance is 2.37 seconds
        #Caching all of the values, preformance is 226 ms

    def test_interpolation_quality(self):
        #Interpolate at the grid bounds and do a numpy.allclose() to see if the spectra match the grid edges

    def test_cache(self):
        pass

    def test_interpolate_keywords(self):
        pass

    #
#* determines if interpolating in T,G,M or T,G,M,A (tri vs quad linear)
#* can determine whether it needs to interpolate in 3 or 4 dimensions
#* caches 8 or 16 (or twice as many) nearby spectra depending on how many T,G,M or T,G,M,A
#* check grid bounds
#* handles edge cases (or uses HDF5 edge handling ability)
#* Set the cache size, and then when the cache is full, pop the first 25% that were loaded.

#What about the weird case of interpolating in alpha, but when the grid is irregular? I think this situation
#would have to be specified by fixing alpha in the lnprob and only using the alpha=0 grid.
#Should we carry the metadata when giving the grid to willie? Or just write out the final values. Or take the
#Average (with weights) of all the other values?