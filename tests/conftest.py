import os
from itertools import product

import numpy as np
import pytest
import scipy.stats as st

from Starfish.emulator import Emulator
from Starfish.grid_tools import download_PHOENIX_models, Instrument, PHOENIXGridInterfaceNoAlpha, \
    HDF5Creator, HDF5Interface
from Starfish.spectrum import DataSpectrum
from Starfish.models import SpectrumParameter, SpectrumModel


@pytest.fixture(scope='session')
def grid_points():
    yield np.array(list(product(
        (6000, 6100, 6200),
        (4.0, 4.5, 5.0),
        (0.0, -0.5, -1.0)
    )))


@pytest.fixture(scope='session')
def PHOENIXModels(grid_points):
    test_base = os.path.dirname(os.path.dirname(__file__))
    outdir = os.path.join(test_base, 'data', 'phoenix')
    download_PHOENIX_models(grid_points, outdir)
    yield outdir


@pytest.fixture(scope='session')
def AlphaPHOENIXModels():
    params = [(6100, 4.5, 0.0, -0.2,)]
    test_base = os.path.dirname(os.path.dirname(__file__))
    outdir = os.path.join(test_base, 'data', 'phoenix')
    download_PHOENIX_models(params, outdir)
    yield outdir


@pytest.fixture(scope='session')
def mock_instrument():
    yield Instrument('Test instrument', FWHM=45.0, wl_range=(1e4, 4e4))


@pytest.fixture(scope='session')
def mock_no_alpha_grid(PHOENIXModels):
    yield PHOENIXGridInterfaceNoAlpha(base=PHOENIXModels)


@pytest.fixture(scope='session')
def mock_creator(mock_no_alpha_grid, mock_instrument, tmpdir_factory, grid_points):
    ranges = np.vstack([np.min(grid_points, 0), np.max(grid_points, 0)]).T
    tmpdir = tmpdir_factory.mktemp('hdf5tests')
    outfile = tmpdir.join('test_grid.hdf5')
    creator = HDF5Creator(mock_no_alpha_grid, filename=outfile, instrument=mock_instrument, wl_range=(2e4, 3e4),
                          ranges=ranges)
    yield creator


@pytest.fixture(scope='session')
def mock_hdf5(mock_creator):
    mock_creator.process_grid()
    yield mock_creator.filename


@pytest.fixture(scope='session')
def mock_hdf5_interface(mock_hdf5):
    yield HDF5Interface(mock_hdf5)


@pytest.fixture
def mock_data():
    wave = np.linspace(1e4, 5e4, 1000)
    peaks = -5 * st.norm.pdf(wave, 2e4, 200)
    peaks += -4 * st.norm.pdf(wave, 2.5e4, 80)
    peaks += -10 * st.norm.pdf(wave, 3e4, 50)
    flux = peaks + np.random.normal(0, 5, size=1000)
    yield wave, flux


@pytest.fixture
def mock_data_spectrum(mock_data):
    wave, flux = mock_data
    yield DataSpectrum(wls=wave, fls=flux)

@pytest.fixture
def mock_emulator(mock_hdf5_interface):
    yield Emulator.from_grid(mock_hdf5_interface)


@pytest.fixture
def mock_trained_emulator(mock_emulator):
    test_base = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(test_base, 'data', 'emu.hdf5')
    if os.path.exists(filename):
        yield Emulator.open(filename)
    else:
        mock_emulator.train()
        mock_emulator.save(filename)


@pytest.fixture
def mock_parameter():
    yield SpectrumParameter(grid_params=[6000., 4.32, -0.2], vz=10., vsini=4.0, logOmega=-0.2, Av=0.3, cheb=[0,
                                                                                                             0.05,
                                                                                                             0.03])

@pytest.fixture
def mock_model(mock_data_spectrum, mock_trained_emulator):
    yield SpectrumModel(mock_trained_emulator, mock_data_spectrum)
