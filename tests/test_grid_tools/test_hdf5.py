import os

import pytest
import h5py

from Starfish.grid_tools import HDF5Creator, HDF5Interface, PHOENIXGridInterfaceNoAlpha
from . import mock_instrument, PHOENIXModels

class TestHDF5Creator:

    @pytest.fixture
    def mock_grid(self, PHOENIXModels):
        yield PHOENIXGridInterfaceNoAlpha(base=PHOENIXModels)

    @pytest.fixture()
    def mock_creator(self, mock_grid, mock_instrument):
        test_base = os.path.dirname(os.path.dirname(__file__))
        outfile = os.path.join(test_base, 'data', 'test_grid.hdf5')
        creator = HDF5Creator(mock_grid, outfile, mock_instrument)
        yield creator

    @pytest.fixture(scope='session')
    def mock_hdf5(self, mock_creator):
        mock_creator.process_grid()
        yield mock_creator.filename

    def test_process_serial(self, mock_creator):
        mock_creator.process_grid(parallel=False)
        assert os.path.exists(mock_creator.filename)

    def test_process_parallel(self, mock_creator):
        mock_creator.process_grid(parallel=True)
        assert os.path.exists(mock_creator.filename)

    def test_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            assert 'wls' in base
            assert 'fls' in base

    def test_no_instrument(self, mock_grid, tmpdir):
        outfile = tmpdir.join('test.hdf5')
        creator = HDF5Creator(mock_grid, outfile, instrument=None)
        creator.process_grid()


class TestHDF5Interface:
    pass