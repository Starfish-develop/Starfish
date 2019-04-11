import os

import pytest
import numpy as np

from Starfish.models import SpectrumModel


class TestSpectrumModel:

    GP = [6200, 4.2, 0]

    @pytest.fixture
    def simple_model(self, mock_emulator, mock_data_spectrum):

        model = SpectrumModel(mock_emulator, mock_data_spectrum, self.GP)
        yield model

    def test_param_dict(self, simple_model):
        assert simple_model['grid_param:0'] == self.GP[0]
        assert simple_model['grid_param:1'] == self.GP[1]
        assert simple_model['grid_param:2'] == self.GP[2]

    def test_grid_params(self, simple_model):
        assert np.all(simple_model.grid_params == self.GP)

    def test_transform(self, simple_model):
        flux, cov = simple_model()
        assert cov.shape == (len(flux), len(flux))
        assert flux.shape == simple_model.wave.shape

    def test_freeze_vsini(self, simple_model):
        simple_model['vsini'] = 84.0
        simple_model.freeze('vsini')
        params = simple_model.get_param_dict()
        assert 'vsini' not in params

    def test_freeze_grid_param(self, simple_model):
        simple_model.freeze('grid_param:2')
        params = simple_model.get_param_dict()
        assert 'grid_param:0' in params
        assert 'grid_param:1' in params
        assert 'grid_param:2' not in params

    def test_freeze_thaw(self, simple_model):
        pre = simple_model['grid_param:1']
        assert pre == 4.2
        simple_model.freeze('grid_param:1')
        assert 'grid_param:1' not in simple_model.get_param_dict()
        simple_model.thaw('grid_param:1')
        assert 'grid_param:1' in simple_model.get_param_dict()
        assert simple_model.grid_params[1] == pre

    def test_get_set_param_dict(self, simple_model):
        P0 = simple_model.get_param_dict()
        simple_model.set_param_dict(P0)
        P1 = simple_model.get_param_dict()
        assert P0 == P1

    def test_set_param_dict_frozen_params(self, simple_model):
        P0 = simple_model.get_param_dict()
        simple_model.freeze('grid_param:2')
        P0['grid_param:2'] = 7
        simple_model.set_param_dict(P0)
        assert simple_model['grid_param:2'] == 0


    def test_get_set_parameters(self, simple_model):
        P0 = simple_model.get_param_vector()
        simple_model.set_param_vector(P0)
        P1 = simple_model.get_param_vector()
        assert np.all(P1 == P0)

    def test_set_wrong_length_param_vector(self, simple_model):
        P0 = simple_model.get_param_vector()
        P1 = np.append(P0, 7)
        with pytest.raises(ValueError):
            simple_model.set_param_vector(P1)

    def test_set_param_vector(self, simple_model):
        P0 = simple_model.get_param_vector()
        P0[2] = 7
        simple_model.set_param_vector(P0)
        assert simple_model['grid_param:2'] == 7


    def test_save_load(self, simple_model, tmpdir):
        path = os.path.join(tmpdir, 'model.json')
        P0 = simple_model.params
        P0_f = simple_model.get_param_dict()
        simple_model.save(path)
        simple_model.load(path)
        assert P0 == simple_model.params
        assert P0_f == simple_model.get_param_dict()
