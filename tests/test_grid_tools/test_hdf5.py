import os

import pytest
import h5py
import numpy as np

from Starfish import config
from Starfish.grid_tools import HDF5Creator, HDF5Interface, PHOENIXGridInterfaceNoAlpha
from . import mock_instrument, PHOENIXModels

class TestHDF5Creator:

    @pytest.fixture(scope='session')
    def mock_grid(self, PHOENIXModels):
        yield PHOENIXGridInterfaceNoAlpha(base=PHOENIXModels)

    @pytest.fixture(scope='session')
    def mock_creator(self, mock_grid, mock_instrument, tmpdir_factory):
        ranges = [
            (6000, 6200),
            (4.0, 5.0),
            (-1.0, 0.0)
        ]
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test_grid.hdf5')
        creator = HDF5Creator(mock_grid, outfile, mock_instrument, wl_range=(2e4, 3e4), ranges=ranges)
        yield creator

    @pytest.fixture(scope='session')
    def mock_hdf5(self, mock_creator):
        mock_creator.process_grid()
        yield mock_creator.filename

    def test_process(self, mock_hdf5):
        assert os.path.exists(mock_hdf5)

    def test_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            assert 'wl' in base
            assert 'pars' in base
            assert 'flux' in base

    def test_wl_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            wave = base['wl']
            assert wave.attrs['air'] == True
            np.testing.assert_approx_equal(wave.attrs['dv'], 7.40, significant=2)
            np.testing.assert_approx_equal(np.min(wave[:]), 2e4 - config.grid['buffer'])
            np.testing.assert_approx_equal(np.max(wave[:]), 3e4 + config.grid['buffer'])

    def test_pars_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            pars = np.array(base['pars'][:])
            assert len(pars) == 27
            np.testing.assert_array_equal(np.min(pars, 0), [6000, 4.0, -1.0])
            np.testing.assert_array_equal(np.max(pars, 0), [6200, 5.0, 0])


    def test_no_instrument(self, mock_grid, tmpdir_factory):
        ranges = [
            (6000, 6200),
            (4.0, 5.0),
            (-1.0, 0.0)
        ]
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test_no_instrument.hdf5')
        creator = HDF5Creator(mock_grid, outfile, instrument=None, wl_range=(2e4, 3e4), ranges=ranges)
        creator.process_grid()


class TestHDF5Interface:
    pass