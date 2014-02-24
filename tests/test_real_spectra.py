import pytest
from StellarSpectra.spectrum import *
from StellarSpectra.grid_tools import *
import numpy as np

class TestRealSpectrum:
    '''
    We've done all of the previous tests using fake spectra. Now try some of the convolution tests using
    a real spectrum and examining what the result is.
    '''
    def setup_class(self):
        hdf5interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        wl = hdf5interface.wl
        self.spec = hdf5interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        self.instrument = Reticon()

    def test_plot_spectrum(self):
        pass

    def test_stellar_convolve(self):
        pass

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