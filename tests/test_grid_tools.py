import pytest
from StellarSpectra.grid_tools import *
import numpy as np
from StellarSpectra.spectrum import create_log_lam_grid
import StellarSpectra.constants as C


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


class TestHDF5Stuffer:
    def setup_class(self):
        import StellarSpectra
        self.rawgrid = PHOENIXGridInterface()
        self.stuffer = HDF5GridStuffer(self.rawgrid, ranges={"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                                                                 "alpha":(0.0, 0.0)}, filename="tests/test.hdf5")

    def test_point_limiter(self):
        ranges = {"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0),
                  "alpha":(0.0, 0.0)}

        points = self.stuffer.points
        #Did the creator object appropriately truncate points out of bounds?
        for key in ranges.keys():
            low, high = ranges[key]
            p = points[key]
            assert np.all(p >= low) and np.all(p <= high), "Points not truncated properly."


    def test_process_flux(self):
        self.stuffer.process_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})

        #requires four parameters
        with pytest.raises(AssertionError) as e:
            self.stuffer.process_flux({"temp":5100, "logg":3.5, "Z":0.0})
        print(e.value)

        #test outside the grid, should be handeled internally
        self.stuffer.process_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":10})


    def test_alpha(self):
        '''Test to make sure that alpha is handled properly, whether specified or not.'''
        alphaHDF5Creator = HDF5GridStuffer(self.rawgrid, ranges={"temp":(5000, 5300), "logg":(3.5, 4.5), "Z":(0.0,0.0)},
                                           filename="test.hdf5")
        alphaHDF5Creator.process_flux({"temp":5100, "logg":3.5, "Z":0.0, "alpha":0.0})


    def test_process_grid(self):
        self.stuffer.process_grid()


class TestHDF5Interface:
    def setup_class(self):
        self.interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")

    #def test_wl(self):
    #    check to see that wl is log-linear spaced
        #wl = self.interface.wl
        #vcs = np.diff(wl)/wl[:-1] * C.c_kms
        #print(vcs)
        #assert np.allclose(vcs, vcs[0]), "wavelength array is not log-lambda spaced."

    def test_bounds(self):
        bounds = self.interface.bounds
        ranges={"temp":(5000, 7000), "logg":(3.5, 5.5), "Z":(-1.0,0.0), "alpha":(0.0, 0.4)}
        for key in C.grid_parameters:
            assert bounds[key] == ranges[key],"Bounds do not match {} != {}".format(bounds[key], ranges[key])

    def test_load_file(self):
        spec = self.interface.load_file({"temp":5100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        print(spec)

    def test_load_bad_file(self):
        with pytest.raises(KeyError) as e:
            self.interface.load_file({"temp":4000, "logg":4.5, "Z": 0.0, "alpha":0.0})
        print(e.value)

    def test_load_flux(self):
        fl = self.interface.load_flux({"temp":5100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        #Try loading just a subset
        self.interface.ind = (10, 20)
        fl = self.interface.load_flux({"temp":5100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        assert len(fl) == 10, "Flux truncation didn't work."



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
        assert interpolator.parameters == {"temp", "logg", "Z"}, "Trilinear truncation didn't work."
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

class TestModelInterpolator:
    def setup_class(self):
    #It is necessary to use a piece of data created on the super computer so we can test interpolation in 4D
        self.hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
    #libraries/PHOENIX_submaster.hd5 should have the following bounds
    #{"temp":(6000, 7000), "logg":(3.5,5.5), "Z":(-1.0,0.0), "alpha":(0.0,0.4)}
        from StellarSpectra.spectrum import DataSpectrum
        self.DataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([21, 22, 23]))

        #TODO: test DataSpectrum with different number of orders, and see how it is truncated.

        self.interpolator = ModelInterpolator(self.hdf5interface, self.DataSpectrum, cache_max=20, cache_dump=10)

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
        interpolator = ModelInterpolator(hdf5interface, self.DataSpectrum)
        assert interpolator.parameters == {"temp", "logg", "Z"}, "Trilinear truncation didn't work."
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1}
        interpolator(parameters)

    def test_trilinear_flag(self):
        hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        interpolator = ModelInterpolator(hdf5interface, self.DataSpectrum, trilinear=True)
        parameters = {"temp":6010, "logg":4.6, "Z":-0.1}
        interpolator(parameters)


    def test_interpolate_bounds(self):
        with pytest.raises(C.InterpolationError) as e:
            parameters = {"temp":4010, "logg":4.6, "Z":-0.1, "alpha":0.1}
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
        intp_flux = self.interpolator(parameters)
        raw_flux = self.hdf5interface.load_flux(parameters)
        assert np.allclose(intp_flux, raw_flux)

    #def test_index_truncate(self):
    #    parameters = {"temp":6000, "logg":4.5, "Z":0.0, "alpha":0.0}
    #    intp_flux = self.interpolator(parameters)
    #    assert len(intp_flux) % 2 == 0, "flux is not power of 2 in length"


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

class TestMasterToFITSIndividual:
    def setup_class(self):
        myHDF5Interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        myInterpolator = Interpolator(myHDF5Interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
                                                                     "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
                                                                     "norm", "PHXVER", "PHXTEFF"])
        self.creator = MasterToFITSIndividual(interpolator=myInterpolator, instrument=KPNO())

    def test_write_to_FITS(self):
        params = {"temp":6100, "logg":4.0, "Z":0.0, "vsini":2}
        self.creator.process_spectrum(params, out_unit="f_nu_log", out_dir="")

class TestMasterToFITSGridProcessor:
    def setup_class(self):
        test_points={"temp":np.arange(6000, 6301, 100), "logg":np.arange(4.0, 4.6, 0.5), "Z":np.arange(-0.5, 0.1, 0.5),
                     "vsini":np.arange(4,9.,2)}
        myHDF5Interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        self.creator = MasterToFITSGridProcessor(interface=myHDF5Interface, instrument=KPNO(), points=test_points,
                                             flux_unit="f_lam", outdir="tests/KPNO/", processes=2)

    def test_param_list(self):
        print(self.creator.param_list)

    def test_process_spectrum_vsini(self):
        params = {"temp":6100, "logg":4.0, "Z":0.0, "alpha":0.0}
        self.creator.process_spectrum_vsini(params)

    def test_out_of_interp_range(self):
        #Will fail silently.
        self.creator.process_spectrum_vsini({"temp":5000, "logg":4.5, "Z":-4.0})

    def test_process_chunk(self):
        chunk = [{"temp":6100, "logg":4.0, "Z":0.0}, {"temp":6000, "logg":4.0, "Z":0.0}, {"temp":6200, "logg":4.0, "Z":0.0}]
        self.creator.process_chunk(chunk)

    def test_f_nu_units(self):
        test_points={"temp":np.arange(6000, 6301, 100), "logg":np.arange(4.0, 4.6, 0.5), "Z":np.arange(-0.5, 0.1, 0.5),
                     "vsini":np.arange(4,9.,2)}
        myHDF5Interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        creator = MasterToFITSGridProcessor(interface=myHDF5Interface, instrument=KPNO(), points=test_points,
                                                 flux_unit="f_nu", outdir="tests/KPNO/", processes=2)
        params = {"temp":6100, "logg":4.0, "Z":0.0, "alpha":0.0}
        creator.process_spectrum_vsini(params)

    def test_process_all(self):
        pass
        #self.creator.process_all()



