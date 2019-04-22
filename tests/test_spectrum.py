import os

import numpy as np

from Starfish.spectrum import DataSpectrum


class TestDataSpectrum:

    def test_masking(self, mock_data_spectrum):
        mask = np.random.randn(*mock_data_spectrum.shape) > 0
        mock_data_spectrum.masks = mask
        assert np.all(mock_data_spectrum.waves == mock_data_spectrum._waves[mask])
        assert np.all(mock_data_spectrum.fluxes ==
                      mock_data_spectrum._fluxes[mask])
        assert np.all(mock_data_spectrum.sigmas ==
                      mock_data_spectrum._sigmas[mask])

    def test_save_load(self, mock_data_spectrum, tmpdir):
        filename = os.path.join(tmpdir, 'data.hdf5')
        mock_data_spectrum.save(filename)
        new_spectrum = DataSpectrum.load(filename)
        assert np.all(new_spectrum.waves == mock_data_spectrum.waves)
        assert np.all(new_spectrum.fluxes == mock_data_spectrum.fluxes)
        assert np.all(new_spectrum.sigmas == mock_data_spectrum.sigmas)
        assert np.all(new_spectrum.masks == mock_data_spectrum.masks)
