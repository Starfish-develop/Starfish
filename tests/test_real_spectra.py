import pytest
from Starfish.spectrum import *
from Starfish.grid_tools import *
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
        self.wl_range = (5160, 5180)

    def test_plot_spectrum(self):
        spec = self.spec.copy()
        plot_spectrum(spec, "tests/plots/basic.png", self.wl_range)

    def test_stellar_convolve(self):
        spec = self.spec.copy()
        spec.stellar_convolve(20.)
        plot_spectrum(spec, "tests/plots/stellar.png", self.wl_range)
        plot_spectrum(spec, "tests/plots/stellar_no_clip.png")


    def test_instrument_convolve(self):
        spec = self.spec.copy()
        spec.instrument_convolve(self.instrument)
        plot_spectrum(spec, "tests/plots/instrument.png")

    def test_instrument_and_stellar_convolve(self):
        spec = self.spec.copy()
        spec.instrument_and_stellar_convolve(self.instrument, 20)
        plot_spectrum(spec, "tests/plots/instrument_and_stellar.png")


    def test_straight_line(self):
        wl_dict = create_log_lam_grid(3000, 13000, min_vc=2/C.c_kms)
        wl = wl_dict.pop("wl")
        fl = wl.copy()
        spec = LogLambdaSpectrum(wl,fl)
        spec.stellar_convolve(50.)
        plot_spectrum(spec, "tests/plots/straight_line.png")
        #Basically, this shows that the up and down behaviour of the Fourier blending is not a problem. It is just
        #blending as if the spectrum wraps. Therefore the two edges are going to match as if it was circular because
        #the stellar kernel is CONVOLVING them TOGETHER, making what was once a 1 pixel transition from max to min now
        #take place over a couple pixels. Thus, there will be edge-effects if the two ends of the spectra aren't very close
