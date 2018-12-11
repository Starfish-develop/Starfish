import os

import pytest

from Starfish import config

default_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')


class TestConfig:

    def test_default_filename(self):
        assert config.filename == default_config

    @pytest.mark.parametrize('key, value', [
        ('plotdir', 'plots/'),
        ('Comments', 'Mid M dwarfs using emulator.\n'),
        ('chunk_ID', 0),
        ('spectrum_ID', 0),
        ('instrument_ID', 0)
    ])
    def test_base_keys(self, key, value):
        assert config[key] == value

    def test_name(self):
        assert config.name == 'default'

    def test_outdir(self):
        assert config.outdir == 'output/'

    def test_grid(self):
        assert isinstance(config.grid, dict)

    @pytest.mark.parametrize('key, value', [
        ('raw_path', '../libraries/raw/CIFIST/'),
        ('hdf5_path', 'grid.hdf5'),
        ('parname', ['temp', 'logg']),
        ('key_name', 't{0:.0f}g{1:.1f}'),
        ('parrange', [[2300, 3700], [4.0, 5.5]]),
        ('wl_range', [6300, 6360]),
        ('buffer', 50)
    ])
    def test_grid_keys(self, key, value):
        assert config.grid[key] == value

    def test_parname(self):
        assert config.parname == config.grid['parname']

    def test_PCA(self):
        assert isinstance(config.PCA, dict)

    @pytest.mark.parametrize('key, value', [
        ('path', 'PCA.hdf5'),
        ('threshold', 0.999),
        ('priors', [[2, 0.0075], [2, 0.75], [2, 0.75]]),
    ])
    def test_PCA_keys(self, key, value):
        assert config.PCA[key] == value

    def test_data(self):
        assert isinstance(config.data, dict)

    @pytest.mark.parametrize('key, value', [
        ('grid_name', 'CIFIST'),
        ('files', ['data.hdf5']),
        ('instruments', ['DCT_DeVeny'])
    ])
    def test_data_keys(self, key, value):
        assert config.data[key] == value

    def test_instruments(self):
        assert config.instruments == config.data['instruments']
