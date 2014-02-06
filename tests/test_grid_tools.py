import pytest
from StellarSpectra.grid_tools import *
import numpy as np



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
        #check to see that it is log-linear spaced
        wl = self.interface.wl
        vcs = np.diff(wl)/wl[:-1] * C.c_kms_air
        print(vcs)
        assert np.allclose(vcs, vcs[0]), "wavelength array is not log-lambda spaced."
        #print(self.interface.flux_name_dict)

    def test_bounds(self):
        print(self.interface.bounds)

    def test_load_file(self):
        self.interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        pass

    def test_load_bad_file(self):
        self.interface.load_file({"temp":5100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        pass


class TestIndexInterpolator:
    def setup_class(self):
        self.interpolator = IndexInterpolator([6000.,6100.,6200.,6300.,6400.])

    def test_interpolate(self):
        ans = self.interpolator(6010)
        print(ans)
        assert ans == ((6000, 6100), (0.9, 0.1))

    def test_interpolate_bounds(self):
        with pytest.raises(InterpolationError) as e:
            self.interpolator(5990)
        print(e.value)

    def test_on_edge(self):
        print(self.interpolator(6100.))


class TestInterpolator:
    def setup_class(self):
        self.hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        #libraries/PHOENIX_submaster.hd5 should have the following bounds
        #{"temp":(6000, 7000), "logg":(3.5,5.5), "Z":(-1.0,0.0), "alpha":(0.0,0.4)}
        self.interpolator = Interpolator(self.hdf5interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
        "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
        "norm", "PHXVER", "PHXTEFF", "BUNNIES"])


    def test_parameters(self):
        print(self.interpolator.parameters)

    def test_trilinear(self):
        hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        hdf5interface.bounds['alpha'] = (0.,0.)
        interpolator = Interpolator(hdf5interface)
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1}
        new_flux = interpolator(parameters)

    def test_interpolate_bounds(self):
        with pytest.raises(InterpolationError) as e:
            parameters = {"temp":5010, "logg":4.6, "Z":-0.1, "alpha":0.1}
            new_flux = self.interpolator(parameters)
        print(e.value)


    def test_quadlinear(self):
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1, "alpha":0.1}
        new_flux = self.interpolator(parameters)
        #Use IPython and  %timeit -n1 -r1 mytest.interpolator({"temp":6010, "logg":5.1, "Z":-0.1, "alpha":-0.1})
        #all uncached performance is 3.89 seconds
        #1/2 uncached performance is 2.37 seconds
        #Caching all of the values, preformance is 226 ms


    def test_cache_similar(self):
        temp = np.random.uniform(6000, 6100, size=3)
        logg = np.random.uniform(3.5, 4.0, size=3)
        Z = np.random.uniform(-0.5, 0.0, size=2)
        alpha = np.random.uniform(0.0, 0.2, size=2)
        names = ["temp", "logg", "Z", "alpha"]
        param_list = [dict(zip(names,param)) for param in itertools.product(temp, logg, Z, alpha)]
        for param in param_list:
            self.interpolator(param)
            print("Cache length", len(self.interpolator.cache))


    def test_cache_purge(self):
        #create a random scattering of possible values, spread throughout a grid cell
        temp = np.random.uniform(6000, 7000, size=3)
        logg = np.random.uniform(3.5, 5.5, size=3)
        Z = np.random.uniform(-1.0, 0.0, size=3)
        alpha = np.random.uniform(0.0, 0.4, size=3)
        names = ["temp", "logg", "Z", "alpha"]
        param_list = [dict(zip(names,param)) for param in itertools.product(temp, logg, Z, alpha)]
        #wait for the cache to purge
        for param in param_list:
            self.interpolator(param)
            print("Cache length", len(self.interpolator.cache))
            print()



    def test_interpolate_keywords(self):
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1, "alpha":0.1}
        new_flux = self.interpolator(parameters)
        print(new_flux)


    def test_interpolation_quality(self):
        #Interpolate at the grid bounds and do a numpy.allclose() to see if the spectra match the grid edges
        #Compare to spectra loaded directly from self.hdf5interface
        parameters = {"temp":6000, "logg":4.5, "Z":0.0, "alpha":0.0}
        intp_spec = self.interpolator(parameters)
        raw_spec = self.hdf5interface.load_file(parameters)
        assert np.allclose(intp_spec.fl, raw_spec.fl)



#What about the weird case of interpolating in alpha, but when the grid is irregular? I think this situation
#would have to be specified by fixing alpha in the lnprob and only using the alpha=0 grid.


class TestInstrument:
    def setup_class(self):
        self.instrument = Reticon()

    def test_initialized(self):
        print(self.instrument)

    def test_log_lam_grid(self):
        print(self.instrument.wl_dict)


class TestMasterToFITSProcessor:
    def setup_class(self):
        test_points={"temp":np.arange(6000, 6251, 250), "logg":np.arange(4.0, 4.6, 0.5), "Z":np.arange(-0.5, 0.1, 0.5), "vsini":np.arange(4,9.,2)}
        myHDF5Interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        myInterpolator = Interpolator(myHDF5Interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
                     "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
                     "norm", "PHXVER", "PHXTEFF"])
        self.creator = MasterToFITSProcessor(interpolator=myInterpolator, instrument=KPNO(), points=test_points, outdir="willie/KPNO/", processes=2)

    def test_param_list(self):
        print("\n", self.creator.param_list, "\n")

    def test_process_all(self):
        self.creator.process_all()

    def test_out_of_interp_range(self):
        self.creator.process_spectrum({"temp":5000, "logg":4.5, "Z":0.0, "vsini":2})
