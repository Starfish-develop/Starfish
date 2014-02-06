import pytest
from StellarSpectra.model import *
from StellarSpectra import grid_tools
import StellarSpectra.constants as C
import matplotlib.pyplot as plt
import numpy as np


testdir = 'tests/plots/'



class TestCreateLogLamGrid:
    def test_log_lam_grid_assert(self):
        wldict = create_log_lam_grid(3000, 13000, min_vc=2/C.c_kms)
        wl = wldict.pop("wl")
        wl_params = wldict

        vcs = np.diff(wl)/wl[:-1] * C.c_kms_air
        print(vcs)
        assert np.allclose(vcs, vcs[0]), "Array must be log-lambda spaced."

    def test_create_log_lam_grid(self):
        print(create_log_lam_grid(min_wl=(0.08,5000)))
        print(create_log_lam_grid(min_vc=2/C.c_kms))
        print(create_log_lam_grid(min_vc=8/C.c_kms))
        with pytest.raises(ValueError) as e:
            create_log_lam_grid()
        print(e.value)


class TestBaseSpectrum:
    def setup_class(self):
        self.spec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)))

    def test_metadata(self):
        print(self.spec.metadata)
        self.spec.add_metadata(("hello","hi"))
        print(self.spec.metadata)

        anotherSpec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)))
        print(anotherSpec.metadata)
        anotherSpec.add_metadata(("hello","hi"))
        print(anotherSpec.metadata)

    def test_air(self):
        pass

class TestBase1DSpectrum(TestBaseSpectrum):
    def setup_class(self):
        self.spec = Base1DSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=3000,))

    def test_calculate_log_lam_grid(self):
        wldict = self.spec.calculate_log_lam_grid()

        wl = wldict.pop("wl")
        wl_params = wldict

        vcs = np.diff(wl)/wl[:-1] * C.c_kms_air
        print(vcs)
        assert np.allclose(vcs, vcs[0]), "Array must be log-lambda spaced."

class TestLogLambdaSpectrum:
    def setup_class(self):
        hdf5interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        wl = hdf5interface.wl
        self.spec = hdf5interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        self.instrument = grid_tools.Reticon()

    def test_load_from_HDF5(self):
        hdf5interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        wl = hdf5interface.wl
        spec = hdf5interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})


    def test_copy(self):
        spec = self.spec.copy()
        spec.wl_raw = np.linspace(0,20)
        print(spec.wl_raw)
        print(self.spec.wl_raw)

    def test_create_from_scratch(self):
        hdf5interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        wl = hdf5interface.wl
        spec = hdf5interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        self.spectrum = LogLambdaSpectrum(wl, spec.fl)
        print("Raw Spectrum", self.spectrum)

    def test_downsample(self):
        self.setup_class()
        self.spec.oversampling = 0.1
        self.spec.downsample()
        print("Downsampled", self.spec)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.spec.wl_raw, self.spec.fl)
        fig.savefig(testdir + "downsample.png")

    def test_instrument_convolve(self):
        self.setup_class()
        print("setup class")
        self.spec.instrument_convolve(self.instrument)
        print("Instrument convolved", self.spec)
        self.spec.instrument_convolve(self.instrument, downsample='yes')
        print("Instrument convolved, downsampled", self.spec)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.spec.wl_raw, self.spec.fl)
        fig.savefig(testdir + "instrument.png")

    def test_stellar_convolve(self):
        self.setup_class()
        self.spec.stellar_convolve(10., downsample='yes')
        print("Stellar convolved", self.spec)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.spec.wl_raw, self.spec.fl)
        fig.savefig(testdir + "stellar.png")

    def test_0_stellar_convolve(self):
        self.setup_class()
        self.spec.stellar_convolve(0., downsample='yes')
        print("Stellar convolved", self.spec)

    def test_neg_stellar_convolve(self):
        self.setup_class()
        self.spec.stellar_convolve(-20., downsample='yes')
        print("Stellar convolved", self.spec)

    def test_instrument_and_stellar_convolve(self):
        self.setup_class()
        self.spec.instrument_and_stellar_convolve(self.instrument, 10., downsample='yes')
        print("Instrument and Stellar convolved", self.spec)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.spec.wl_raw, self.spec.fl)
        fig.savefig(testdir + "instrument_and_stellar.png")

    def test_0_instrument_and_stellar_convolve(self):
        self.setup_class()
        self.spec.instrument_and_stellar_convolve(self.instrument, 0., downsample='yes')
        print("Instrument and Stellar convolved", self.spec)

    def test_neg_instrument_and_stellar_convolve(self):
        self.setup_class()
        self.spec.instrument_and_stellar_convolve(self.instrument, 0., downsample='yes')
        print("Instrument and Stellar convolved", self.spec)

    def test_straight_line(self):
        print("Straight line")
        wldict = create_log_lam_grid(3000, 13000, min_vc=2/C.c_kms)
        wl = wldict.pop("wl")
        fl = wl.copy()
        self.spectrum = LogLambdaSpectrum(wl,fl)
        self.spectrum.stellar_convolve(50., downsample='yes')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.spectrum.wl_raw, self.spectrum.fl)
        fig.savefig(testdir + "line_stellar.png")
        #Basically, this shows that the up and down behaviour of the Fourier blending is not a problem. It is just
        #blending as if the spectrum wraps. Therefore the two edges are going to match as if it was circular.



#These multiple commands should not repeat the same shortening, but they seem to be.