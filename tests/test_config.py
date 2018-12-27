import os


import pytest
import yaml

from Starfish import config, DEFAULT_CONFIG_FILE
from Starfish._config import Config


class TestConfig:

    @pytest.fixture
    def test_config(self):
        base_dir = os.path.dirname(__file__)
        filename = os.path.join(base_dir, 'data', 'test_config.yaml')
        yield Config(filename)

    def test_default_filename(self):
        default_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Starfish', 'config.yaml')
        assert DEFAULT_CONFIG_FILE == default_config

    @pytest.mark.parametrize('key, value', [
        ('plotdir', 'plots/'),
        ('Comments', 'Mid M dwarfs using emulator.\n'),
        ('chunk_ID', 0),
        ('spectrum_ID', 0),
        ('instrument_ID', 0),
        ('specfmt', 's{}_o{}'),
    ])
    def test_base_keys(self, test_config, key, value):
        assert test_config[key] == value

    def test_base_dots(self):
        assert config.grid == config['grid']
        assert config.PCA == config['PCA']
        assert config.data == config['data']
        assert config.name == config['name']
        assert config.outdir == config['outdir']

    def test_name(self):
        assert config.name == 'default'

    def test_outdir(self):
        assert config.outdir == 'output/'

    @pytest.mark.parametrize('key, value', [
        ('raw_path', '../libraries/raw/PHOENIX/'),
        ('hdf5_path', 'grid.hdf5'),
        ('parname', ['Teff', 'logg', 'Z']),
        ('key_name', 'T{0:.0f}_g{1:.1f}_Z{2:.2f}'),
        ('parrange', [[6000, 6200], [4.0, 5.0], [-1.0, 0.0]]),
        ('wl_range', [2e4, 3e4]),
        ('buffer', 50)
    ])
    def test_grid_keys(self, key, value):
        assert config.grid[key] == value

    @pytest.mark.parametrize('key, value', [
        ('path', 'PCA.hdf5'),
        ('threshold', 0.999),
        ('priors', [[2, 0.0075], [2, 0.75], [2, 0.75]]),
    ])
    def test_PCA_keys(self, key, value):
        assert config.PCA[key] == value

    @pytest.mark.parametrize('key, value', [
        ('grid_name', 'PHOENIX'),
        ('files', ['data.hdf5']),
        ('instruments', ['DCT_DeVeny'])
    ])
    def test_data_keys(self, key, value):
        assert config.data[key] == value

    def test_change_file(self):
        previous = config.name
        base_dir = os.path.dirname(__file__)
        filename = os.path.join(base_dir, 'data', 'test_config.yaml')
        config.change_file(filename)
        assert config.name != previous
        assert config._path == filename

    def test_set_attr_fail_on_default(self):
        with pytest.raises(RuntimeError):
            config.name = 'Stephen King'

    def test_set_base_attr(self, test_config):
        previous = test_config.name
        test_config.name = 'new name'
        assert test_config.name == 'new name'
        test_config.name = previous
        assert test_config.name == previous

    def test_set_non_base_attr(self, test_config):
        old_path = test_config.PCA['path']
        test_config.PCA['path'] = 'testpath.hdf5'
        assert test_config.PCA['path'] == 'testpath.hdf5'
        test_config.PCA['path'] = old_path
        assert test_config.PCA['path'] == old_path

    def test_copy_config(self, tmpdir):
        assert not os.path.exists(tmpdir.join('config.yaml'))
        config.copy_file(tmpdir)
        assert os.path.exists(tmpdir.join('config.yaml'))

    def test_lazy_loading(self, test_config):
        id = test_config.cheb_degree
        base = test_config._config
        base['cheb_degree'] = id + 1
        with open(test_config._path, 'w') as f:
            yaml.safe_dump(base, f)

        assert test_config.cheb_degree == id + 1
        test_config.cheb_degree = id
        assert test_config.cheb_degree == id
