import pytest
from StellarSpectra.model import *
from StellarSpectra import grid_tools
import StellarSpectra.constants as C



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

    def test_create_from_scratch(self):
        hdf5interface = grid_tools.HDF5Interface("libraries/PHOENIX_submaster.hdf5")
        wl = hdf5interface.wl
        spec = hdf5interface.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})
        self.spectrum = LogLambdaSpectrum(wl, spec.fl)
        print(self.spectrum.metadata)
        pass

    def test_downsample(self):
        self.spec.oversampling = 0.1
        self.spec.downsample()
        print(self.spec.metadata)

    def test_instrument_convolve(self):
        self.spec.instrument_convolve(self.instrument, downsample=False)


    def test_stellar_convolve(self):
        pass