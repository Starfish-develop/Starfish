import os
import types
from itertools import product

import h5py
import numpy as np
import pytest

from Starfish.grid_tools import HDF5Creator, HDF5Interface


class TestHDF5Creator:

    def test_process(self, mock_hdf5):
        assert os.path.exists(mock_hdf5)

    def test_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            assert 'wl' in base
            assert 'grid_points' in base
            assert 'flux' in base

    def test_wl_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            wave = base['wl']
            assert wave.attrs['air'] == True
            np.testing.assert_approx_equal(
                wave.attrs['dv'], 7.40, significant=2)
            np.testing.assert_approx_equal(np.min(wave[:]), 2e4 - 50)
            np.testing.assert_approx_equal(np.max(wave[:]), 3e4 + 50)

    def test_pars_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            pars = np.array(base['grid_points'][:])
            assert len(pars) == 27
            np.testing.assert_array_equal(np.min(pars, 0), [6000, 4.0, -1.0])
            np.testing.assert_array_equal(np.max(pars, 0), [6200, 5.0, 0])

    def test_flux_contents(self, mock_hdf5):
        with h5py.File(mock_hdf5) as base:
            flux = base['flux']
            assert flux.attrs['units'] == 'erg/s/cm^2/cm'
            assert len(flux) == len(base['grid_points'])
            assert flux.attrs['key_name'] == 'Z{2:}__lte{0:0>5.0f}-{1:.2f}{2:}.PHOENIX-ACES-AGSS-COND-2011-HiRes'
            name = flux.attrs['key_name'].format(6000, 4.5, -0.5)
            assert name in flux
            assert flux[name].shape == base['wl'].shape

    def test_no_instrument(self, mock_no_alpha_grid, tmpdir_factory, grid_points):
        ranges = np.vstack([np.min(grid_points, 0), np.max(grid_points, 0)]).T
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test_no_instrument.hdf5')
        creator = HDF5Creator(mock_no_alpha_grid, filename=outfile, instrument=None, wl_range=(2e4, 3e4),
                              ranges=ranges)
        creator.process_grid()

    @pytest.mark.parametrize('wl_range', [
        (2e4, 3e3),
        # Should fail because 2e4 will be clipped up to minimum instrument wavelength
        (2e3, 3e3),
        (4e4, 2.4e4),
        (5e4, 6e4)
    ])
    def test_invalid_wavelengths(self, wl_range, mock_no_alpha_grid, mock_instrument, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test_no_instrument.hdf5')
        with pytest.raises(ValueError):
            HDF5Creator(mock_no_alpha_grid, filename=outfile,
                        instrument=mock_instrument, wl_range=wl_range)

    @pytest.mark.parametrize('wl_range, wl_expected', [
        [(5e3, 2e4), (1e4, 2e4 + 50)],
        [(1e4, 2e4), (1e4, 2e4 + 50)],
        [(1e4, 4e4), (1e4, 4e4)],
        [(2e4, 4e4), (2e4 - 50, 4e4)],
        [(2e4, 6e4), (2e4 - 50, 4e4)],
    ])
    def test_wavelength_truncation(self, wl_range, wl_expected, mock_no_alpha_grid, mock_instrument, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test_no_instrument.hdf5')
        creator = HDF5Creator(mock_no_alpha_grid, filename=outfile,
                              instrument=mock_instrument, wl_range=wl_range)
        np.testing.assert_array_almost_equal_nulp(
            creator.wl_final[0], wl_expected[0], 3)
        np.testing.assert_array_almost_equal_nulp(
            creator.wl_final[-1], wl_expected[-1], 3)

    def test_no_wl_truncation(self, mock_no_alpha_grid, mock_instrument, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test_no_instrument.hdf5')
        creator = HDF5Creator(mock_no_alpha_grid,
                              filename=outfile, instrument=None, wl_range=None)
        np.testing.assert_array_almost_equal_nulp(
            creator.wl_final, mock_no_alpha_grid.wl[1:-1])

    def test_par_truncation(self, grid_points, mock_no_alpha_grid, mock_instrument, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test.hdf5')
        ranges = [
            (6000, 6200),
            (4.0, 5.0),
            (-1.0, 0.0)
        ]
        creator = HDF5Creator(mock_no_alpha_grid,
                              filename=outfile, instrument=mock_instrument, ranges=ranges)
        for cpars, gpars in zip(creator.points, grid_points.T):
            np.testing.assert_allclose(cpars, np.unique(gpars))

    def test_no_par_truncation(self, mock_no_alpha_grid, mock_instrument, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('hdf5tests')
        outfile = tmpdir.join('test_no_instrument.hdf5')
        creator = HDF5Creator(
            mock_no_alpha_grid, filename=outfile, instrument=mock_instrument, ranges=None)
        for cpars, gpars in zip(creator.points, np.asarray(mock_no_alpha_grid.points).T):
            np.testing.assert_allclose(cpars, np.unique(gpars))


class TestHDF5Interface:

    def test_filename(self, mock_hdf5_interface, mock_hdf5):
        assert mock_hdf5_interface.filename == mock_hdf5

    @pytest.mark.parametrize('param', [
        (6000, 4.5, 0),
        (6100, 4.0, -1.0),
        (6200, 5.0, -0.5)
    ])
    def test_load_flux(self, param, mock_hdf5_interface):
        flux, hdr = mock_hdf5_interface.load_flux(param, header=True)
        assert hdr['PHXTEFF'] == param[0]
        assert hdr['PHXLOGG'] == param[1]
        assert hdr['PHXM_H'] == param[2]

    def test_load_flux_no_hdr(self, mock_hdf5_interface):
        flux = mock_hdf5_interface.load_flux((6000, 4.5, 0))
        assert isinstance(flux, np.ndarray)

    def test_fluxes(self, mock_hdf5_interface, grid_points):
        fluxes = mock_hdf5_interface.fluxes
        assert isinstance(fluxes, types.GeneratorType)
        assert len(list(fluxes)) == len(list(grid_points))
