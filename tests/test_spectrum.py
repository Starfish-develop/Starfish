import pytest
from StellarSpectra.spectrum import *
import numpy as np
import tempfile
import os

@pytest.fixture()
def cleandir():
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)

class TestBaseSpectrum:
    def setup_class(self):
        #Initialize the spectrum with random flux
        self.spec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)), metadata={"type":"test spectrum"})

    def test_initialize(self):
        #Initialize with mis-matched wl and flux arrays
        with pytest.raises(AssertionError) as e:
            spec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(2000,)))
        print(e.value)

    def test_metadata(self):
        print(self.spec)
        assert self.spec.metadata["type"] == "test spectrum"
        assert self.spec.metadata["air"] == True
        assert self.spec.metadata["unit"] == "f_lam"

    def test_air(self):
        spec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)))
        assert spec.metadata["air"] == True

        spec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)), air=False)
        assert spec.metadata["air"] == False

    def test_convert_units(self):
        spec = self.spec.copy() #Backup spectrum to compare
        assert spec.metadata['unit'] == "f_lam"
        #Test that no conversion takes place
        spec.convert_units("f_lam")
        assert spec.metadata['unit'] == "f_lam"

        #Convert from f_lam to f_nu
        spec.convert_units("f_nu")
        assert spec.metadata['unit'] == "f_nu"
        #check to make sure it was transferred correctly
        assert np.allclose(spec.fl, self.spec.wl**2 * self.spec.fl/C.c_ang)

        #Now convert back from f_nu to f_lam
        assert spec.metadata['unit'] == "f_nu"
        spec.convert_units("f_lam")
        assert spec.metadata['unit'] == "f_lam"
        #check to make sure everything is back as it belongs
        assert np.allclose(spec.fl, self.spec.fl)


    @pytest.mark.usefixtures("cleandir")
    def test_save(self):
        self.spec.save("Spectrum.npy")
        wl, fl = np.load("Spectrum.npy")
        assert np.allclose(wl, self.spec.wl)
        assert np.allclose(fl, self.spec.fl)

    def test_copy(self):
        newspec = self.spec.copy()
        assert np.allclose(newspec.wl, self.spec.wl)
        assert np.allclose(newspec.fl, self.spec.fl)
        assert newspec.metadata == self.spec.metadata

    def test_str(self):
        print(self.spec)


class Test_create_log_lam_grid:
    '''
    Test the creation of a log lam spaced grid.
    '''
    def setup_class(self):
        self.wl_dict = create_log_lam_grid(3000, 10000, min_vc=2/C.c_kms)

    def test_starting_ranges(self):
        #Swap starting ranges
        with pytest.raises(AssertionError) as e:
            create_log_lam_grid(10000, 2000, min_vc=2)
        print(e.value)

    def test_min_vc_min_wl(self):
        min_vc = 2/C.c_kms #km/s

        #at lam = 5000 ang, what would the delta lam be that corresponds to 2 km/s?
        lam = 5000
        delta_lam = min_vc * lam
        print("2 km/s delta_lam is {:.4f} ang at {:.2f} ang".format(delta_lam, lam))

        #See that the routine raises an error when neither min_vc nor (delta_lam, lam) specified
        with pytest.raises(ValueError) as e:
            create_log_lam_grid(2000, 10000)
        print(e.value)

        #Make sure it branches correctly
        #make delta lam *slightly* bigger than min_vc, so min_vc is chosen
        wl_dict1 = create_log_lam_grid(2000, 10000, min_vc=min_vc, min_wl=(delta_lam + 0.002, lam))

        #make min_vc slightly larger, so delta_lam is chosen
        wl_dict2 = create_log_lam_grid(2000, 10000, min_vc=min_vc* 1.05, min_wl=(delta_lam, lam))

        #These two should have the same wavelength and header values
        assert np.allclose(wl_dict1['wl'], wl_dict2['wl']), "Wavelength arrays do not match"
        for kw in log_lam_kws:
            assert wl_dict1[kw] == wl_dict2[kw], \
                "{} does not match between the two spectra. {} != {}".format(kw, wl_dict1[kw], wl_dict2[kw])


class TestBase1DSpectrum(TestBaseSpectrum):
    def setup_class(self):
        #Initialize the spectrum with random flux
        self.spec = Base1DSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)), metadata={"type":"test spectrum"})

    def test_initialize(self):
        #Initialize with mis-matched wl and flux arrays
        with pytest.raises(AssertionError) as e:
            spec = Base1DSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(2000,)))
        print(e.value)

        #Now try with 2-D arrays
        with pytest.raises(AssertionError) as e:
            wl = np.linspace(5000, 5100, num=3000)
            wl.shape = (3, -1)
            spec = Base1DSpectrum(wl, np.ones_like(wl))
        print(e.value)

    def test_calculate_log_lam_grid(self):
        wl_dict = self.spec.calculate_log_lam_grid()

        wl = self.spec.wl
        dif = np.diff(wl)
        min_wl = np.min(dif)
        wl_at_min = wl[np.argmin(dif)]

        #Make sure that the header keywords reflect what you would actually expect
        min_vc = 10**wl_dict['CDELT1'] - 1

        assert min_vc <= min_wl/wl_at_min*1.00001, "Velocity sampling not respected. {} larger than {}".format(min_vc, min_wl/wl_at_min)

    def test_resample_to_grid(self):
        #Try giving it a range that is too big
        spec = self.spec.copy()
        with pytest.raises(ValueError) as e:
            spec.resample_to_grid(np.linspace(4000, 7000))
        print(e.value)

    def test_integrate(self):
        #Do spline interpolation on easy spectrum, that way we know the results
        wl = np.arange(1, 100)
        f = (C.h * C.c_ang)/wl #this is inverse counts/ang
        spec = Base1DSpectrum(wl, f)

        spec.resample_to_grid(np.arange(2, 10.1, 0.5), integrate=True)
        #Each pixel in spec.wl should be equal to 100 at the end
        assert np.allclose(spec.fl, 100.),"Integration failed"

        #Try integrating twice
        spec = self.spec.copy()
        spec.resample_to_grid(np.linspace(5010, 5020), integrate=True)

        with pytest.raises(AssertionError) as e:
            spec.resample_to_grid(np.linspace(5010, 5020), integrate=True)
        print(e.value)


class TestLogLambdaSpectrum(TestBase1DSpectrum):
    def setup_class(self):
        wl_dict = create_log_lam_grid(3000, 10000, min_vc=2/C.c_kms)
        wl = wl_dict.pop("wl")
        wl_dict.update({"type":"test spectrum"})
        self.spec = LogLambdaSpectrum(wl, np.ones_like(wl), metadata=wl_dict)

    def test_initialize(self):
        #Try to initialize with non-power of 2 array
        wl_dict = create_log_lam_grid(3000, 10000, min_vc=2/C.c_kms)
        wl = wl_dict.pop("wl")
        wl = wl[:-101]
        fl = np.ones_like(wl)
        spec = LogLambdaSpectrum(wl, np.ones_like(wl))

        # test_bad_metadata
        with pytest.raises(AssertionError) as e:
            spec = LogLambdaSpectrum(wl, np.ones_like(wl), metadata=wl_dict)
        print(e.value)


    def test_resample_to_grid(self):
        #Try giving it a range that is too big
        spec = self.spec.copy()
        wl_dict = create_log_lam_grid(2000, 12000, min_vc=2/C.c_kms)
        with pytest.raises(ValueError) as e:
            spec.resample_to_grid(wl_dict)
        print(e.value)

    def test_integrate(self):

        #Do spline interpolation on easy spectrum, that way we know the results
        wl_dict = create_log_lam_grid(5000, 6000, min_vc=2/C.c_kms)
        wl = wl_dict.pop('wl')

        f = (C.h * C.c_ang)/wl #this is inverse counts/ang
        spec = LogLambdaSpectrum(wl, f, metadata=wl_dict)

        new_wl_dict = create_log_lam_grid(5500, 5600, min_vc=8/C.c_kms)

        spec.resample_to_grid(new_wl_dict, integrate=True)
        #It's hard to check this, because now the wl grid is differently spaced

        #Try integrating twice
        spec = self.spec.copy()
        spec.resample_to_grid(new_wl_dict, integrate=True)

        with pytest.raises(AssertionError) as e:
            spec.resample_to_grid(new_wl_dict, integrate=True)
        print(e.value)

    def test_convolve_with_gaussian(self):
        self.setup_class()
        self.spec.convolve_with_gaussian(14.4)

    def test_stellar_convolve(self):
        self.setup_class()
        self.spec.stellar_convolve(10.)

        #Try to do it twice
        with pytest.raises(AssertionError) as e:
            self.spec.stellar_convolve(10)
        print(e.value)

        self.setup_class()
        self.spec.stellar_convolve(-1.)

    def test_instrument_convolve(self):
        self.setup_class()

        from StellarSpectra.grid_tools import Reticon
        instrument = Reticon()

        self.spec.instrument_convolve(instrument)

        #try to do it twice
        with pytest.raises(AssertionError) as e:
            self.spec.instrument_convolve(instrument)
        print(e.value)

    def test_instrument_convolve_integrate(self):
        self.setup_class()

        from StellarSpectra.grid_tools import Reticon
        instrument = Reticon()

        self.spec.instrument_convolve(instrument, integrate=True)

    def test_instrument_and_stellar_convolve(self):
        self.setup_class()
        from StellarSpectra.grid_tools import Reticon
        instrument = Reticon()
        self.spec.instrument_and_stellar_convolve(instrument, 10.)



class TestRealSpectrum:
    '''
    We've done all of the previous tests using fake spectra. Now try some of the convolution tests using
    a real spectrum and examining what the result is.
    '''
    def setup_class(self):
        pass

        #hdf5interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        #wl = hdf5interface.wl
        #self.spec = hdf5interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        #self.instrument = grid_tools.Reticon()

        #

        #

        #    print("Instrument and Stellar convolved", self.spec)
        #    fig = plt.figure()
        #    ax = fig.add_subplot(111)
        #    ax.plot(self.spec.wl_raw, self.spec.fl)
        #    fig.savefig(testdir + "instrument_and_stellar.png")
        #
        #def test_0_instrument_and_stellar_convolve(self):
        #    self.setup_class()
        #    self.spec.instrument_and_stellar_convolve(self.instrument, 0., downsample='yes')
        #    print("Instrument and Stellar convolved", self.spec)
        #
        #def test_neg_instrument_and_stellar_convolve(self):
        #    self.setup_class()
        #    self.spec.instrument_and_stellar_convolve(self.instrument, 0., downsample='yes')
        #    print("Instrument and Stellar convolved", self.spec)
        #
        #def test_straight_line(self):
        #    print("Straight line")
        #    wldict = create_log_lam_grid(3000, 13000, min_vc=2/C.c_kms)
        #    wl = wldict.pop("wl")
        #    fl = wl.copy()
        #    self.spectrum = LogLambdaSpectrum(wl,fl)
        #    self.spectrum.stellar_convolve(50., downsample='yes')
        #    fig = plt.figure()
        #    ax = fig.add_subplot(111)
        #    ax.plot(self.spectrum.wl_raw, self.spectrum.fl)
        #    fig.savefig(testdir + "line_stellar.png")
        #    #Basically, this shows that the up and down behaviour of the Fourier blending is not a problem. It is just
        #    #blending as if the spectrum wraps. Therefore the two edges are going to match as if it was circular.