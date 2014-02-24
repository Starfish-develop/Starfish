import pytest
from StellarSpectra.grid_tools import *
import numpy as np
from StellarSpectra.spectrum import create_log_lam_grid


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
        with pytest.raises(GridError) as e:
            self.rawgrid.check_params({"temp":5100, "logg":3.5, "bunny":0.0})
        print(e.value)

    def test_check_params_outbouts(self):
        with pytest.raises(GridError) as e:
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
        with pytest.raises(GridError) as e:
            self.rawgrid.load_file({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.2})
        print(e.value)

    def test_load_alpha(self):
        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0})

        self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})

    def test_load_file_metadata(self):
        spec = self.rawgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0})
        print(spec.metadata)

    def test_bad_base(self):
        #Set a different base location, should raise an error because on this machine there is not file.
        with pytest.raises(GridError) as e:
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


class TestHDF5Creator:
    def setup_class(self):
        import StellarSpectra
        self.wl_dict = create_log_lam_grid(5000, 6000, min_vc=2/C.c_kms)
        self.rawgrid = PHOENIXGridInterface()
        self.HDF5Creator = HDF5GridCreator(self.rawgrid, ranges={"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                                            "alpha":(0.0, 0.0)}, filename="tests/test.hdf5", wl_dict=self.wl_dict)

    def test_point_limiter(self):
        ranges = {"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                "alpha":(0.0, 0.0)}

        points = self.HDF5Creator.points
        #Did the creator object appropriately truncate points out of bounds?
        for key in ranges.keys():
            low, high = ranges[key]
            p = points[key]
            assert np.all(p >= low) and np.all(p <= high), "Points not truncated properly."


    def test_process_flux(self):
        self.HDF5Creator.process_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})

        #requires four parameters
        with pytest.raises(AssertionError) as e:
            self.HDF5Creator.process_flux({"temp":5100, "logg":3.5, "Z":0.0})
        print(e.value)

        #test outside the grid, should be handeled internally
        self.HDF5Creator.process_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":10})


    def test_alpha(self):
        '''Test to make sure that alpha is handled properly, whether specified or not.'''
        alphaHDF5Creator = HDF5GridCreator(self.rawgrid, ranges={"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0)},
                                           filename="test.hdf5", wl_dict=self.wl_dict, chunksize=20)
        alphaHDF5Creator.process_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})


    def test_process_grid(self):
        self.HDF5Creator.process_grid()

    def test_parallel(self):
        HDF5Creator = HDF5GridCreator(self.rawgrid, ranges={"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                        "alpha":(0.0, 0.0)}, filename="tests/parallel_test.hdf5", wl_dict=self.wl_dict, nprocesses=2)
        HDF5Creator.process_grid()

class TestHDF5Interface:
    def setup_class(self):
        self.interface = HDF5Interface("tests/test.hdf5")

    def test_wl(self):
        #check to see that wl is log-linear spaced
        wl = self.interface.wl
        vcs = np.diff(wl)/wl[:-1] * C.c_kms
        print(vcs)
        assert np.allclose(vcs, vcs[0]), "wavelength array is not log-lambda spaced."

    def test_bounds(self):
        bounds = self.interface.bounds
        ranges={"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0), "alpha":(0.0, 0.0)}
        for key in grid_parameters:
            assert bounds[key] == ranges[key],"Bounds do not match {} != {}".format(bounds[key], ranges[key])

    def test_load_file(self):
        spec = self.interface.load_file({"temp":5100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        print(spec)

    def test_load_bad_file(self):
        with pytest.raises(KeyError) as e:
            self.interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        print(e.value)


class TestIndexInterpolator:
    def setup_class(self):
        self.interpolator = IndexInterpolator([6000.,6100.,6200.,6300.,6400.])

    def test_interpolate(self):
        ans = self.interpolator(6010)
        assert ans == ((6000, 6100), (0.9, 0.1))

    def test_interpolate_bounds(self):
        with pytest.raises(InterpolationError) as e:
            self.interpolator(5990)
        print(e.value)

    def test_on_edge(self):
        print(self.interpolator(6100.))


class TestInterpolator:
    def setup_class(self):
        #It is necessary to use a piece of data created on the super computer so we can test interpolation in 4D
        self.hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        #libraries/PHOENIX_submaster.hd5 should have the following bounds
        #{"temp":(6000, 7000), "logg":(3.5,5.5), "Z":(-1.0,0.0), "alpha":(0.0,0.4)}
        self.interpolator = Interpolator(self.hdf5interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
        "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
        "norm", "PHXVER", "PHXTEFF", "BUNNIES"], cache_max=20, cache_dump=10)

    def test_quadlinear(self):
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1, "alpha":0.1}
        self.interpolator(parameters)

        #Use IPython and  %timeit -n1 -r1 mytest.interpolator({"temp":6010, "logg":5.1, "Z":-0.1, "alpha":-0.1})
        #all uncached performance is 3.89 seconds
        #1/2 uncached performance is 2.37 seconds
        #Caching all of the values, preformance is 226 ms

    def test_trilinear(self):
        hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        hdf5interface.bounds['alpha'] = (0.,0.) #manually set alpha range to 0
        interpolator = Interpolator(hdf5interface)
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1}
        interpolator(parameters)

    def test_interpolate_bounds(self):
        with pytest.raises(InterpolationError) as e:
            parameters = {"temp":5010, "logg":4.6, "Z":-0.1, "alpha":0.1}
            new_flux = self.interpolator(parameters)
        print(e.value)

    def test_cache_similar(self):
        temp = np.random.uniform(6000, 6100, size=3)
        logg = np.random.uniform(3.5, 4.0, size=3)
        Z = np.random.uniform(-0.5, 0.0, size=2)
        alpha = np.random.uniform(0.0, 0.2, size=2)
        names = ["temp", "logg", "Z", "alpha"]
        param_list = [dict(zip(names,param)) for param in itertools.product(temp, logg, Z, alpha)]
        for param in param_list:
            print("Cache length", len(self.interpolator.cache))
            self.interpolator(param)

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
            print("Cache length", len(self.interpolator.cache))
            self.interpolator(param)

    def test_interpolation_quality(self):
        '''
        Interpolate at the grid bounds and do a numpy.allclose() to see if the spectra match the grid edges
        '''
        #Compare to spectra loaded directly from self.hdf5interface
        parameters = {"temp":6000, "logg":4.5, "Z":0.0, "alpha":0.0}
        intp_spec = self.interpolator(parameters)
        raw_spec = self.hdf5interface.load_file(parameters)
        assert np.allclose(intp_spec.fl, raw_spec.fl)


#What about the weird case of interpolating in alpha, but when the grid is irregular? I think this situation
#would have to be specified by fixing alpha in the lnprob and only using the alpha=0 grid.


class TestInstrument:
    def test_TRES(self):
        instrument = TRES()
        print(instrument)
        print(instrument.wl_dict)

    def test_Reticon(self):
        instrument = Reticon()
        print(instrument)
        print(instrument.wl_dict)

    def test_KPNO(self):
        instrument = KPNO()
        print(instrument)
        print(instrument.wl_dict)

class TestMasterToFITSProcessor:
    def setup_class(self):
        test_points={"temp":np.arange(6000, 6251, 250), "logg":np.arange(4.0, 4.6, 0.5), "Z":np.arange(-0.5, 0.1, 0.5), "vsini":np.arange(4,9.,2)}
        myHDF5Interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        myInterpolator = Interpolator(myHDF5Interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
                     "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
                     "norm", "PHXVER", "PHXTEFF"])
        self.creator = MasterToFITSProcessor(interpolator=myInterpolator, instrument=KPNO(), points=test_points,
                                             flux_unit="f_lam", outdir="tests/KPNO/", processes=2)

    def test_param_list(self):
        print(self.creator.param_list)

    def test_process_file(self):
        pass

    def test_write_to_fits(self):
        pass

    def test_process_all(self):
        self.creator.process_all()

    def test_out_of_interp_range(self):
        self.creator.process_spectrum({"temp":5000, "logg":4.5, "Z":-4.0})

