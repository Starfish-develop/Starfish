import os
from itertools import product

import numpy as np
import pytest
import scipy.stats as st

from Starfish.emulator import Emulator
from Starfish.grid_tools import download_PHOENIX_models, Instrument, PHOENIXGridInterfaceNoAlpha, \
    HDF5Creator, HDF5Interface
from Starfish.models import SpectrumModel
from Starfish.models.transforms import resample
from Starfish.spectrum import DataSpectrum
from Starfish.utils import create_log_lam_grid

test_base = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def grid_points():
    yield np.array(list(product(
        (6000, 6100, 6200),
        (4.0, 4.5, 5.0),
        (0.0, -0.5, -1.0)
    )))


@pytest.fixture(scope='session')
def PHOENIXModels(grid_points):
    outdir = os.path.join(test_base, 'data', 'phoenix')
    download_PHOENIX_models(grid_points, outdir)
    yield outdir


@pytest.fixture(scope='session')
def AlphaPHOENIXModels():
    params = [(6100, 4.5, 0.0, -0.2,)]
    outdir = os.path.join(test_base, 'data', 'phoenix')
    download_PHOENIX_models(params, outdir)
    yield outdir


@pytest.fixture(scope='session')
def mock_instrument():
    yield Instrument('Test instrument', FWHM=45.0, wl_range=(1e4, 4e4))


@pytest.fixture(scope='session')
def mock_no_alpha_grid(PHOENIXModels):
    yield PHOENIXGridInterfaceNoAlpha(path=PHOENIXModels)


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
def mock_data(mock_hdf5_interface):
    wave = mock_hdf5_interface.wl
    new_wave = create_log_lam_grid(1e3, wave.min(), wave.max())['wl']
    flux = resample(wave, next(mock_hdf5_interface.fluxes), new_wave)
    yield new_wave, flux


@pytest.fixture
def mock_data_spectrum(mock_data):
    wave, flux = mock_data
    sigs = np.random.randn(len(flux))
    yield DataSpectrum(waves=wave, fluxes=flux, sigmas=sigs)


@pytest.fixture
def mock_emulator(mock_hdf5_interface):
    yield Emulator.from_grid(mock_hdf5_interface)


@pytest.fixture
def mock_trained_emulator(mock_emulator):
    filename = os.path.join(test_base, 'data', 'emu.hdf5')
    if os.path.exists(filename):
        yield Emulator.load(filename)
    else:
        mock_emulator.train()
        mock_emulator.save(filename)


@pytest.fixture
def mock_model(mock_data_spectrum, mock_trained_emulator):
    global_params = {
        'log_amp': 1,
        'log_ls': 1,
    }
    local_params = [
        {
            'mu': 1e4,
            'log_amp': 2,
            'log_sigma': 2,
        },
        {
            'mu': 1.3e4,
            'log_amp': 1.5,
            'log_sigma': 2
        }
    ]
    yield SpectrumModel(mock_trained_emulator, grid_params=[6000, 4.0, 0.0], data=mock_data_spectrum, vz=0, Av=0,
                        log_scale=-10, vsini=30, glob=global_params, local=local_params)
