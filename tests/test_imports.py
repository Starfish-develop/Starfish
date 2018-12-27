"""
This file should be used as a check to see if the RTD build will fail, since it
imports all of the code but doesn't run any of it. This is important because
python 3.5 is very buggy for our code but RTD builds using python 3.5.2

This is janky code, don't reproduce this anywhere except this test file, please.
"""
import pytest

class TestImports:

    def test_base(self):
        import Starfish
        exec('from Starfish import *')

    @pytest.mark.parametrize('module', [
        '_config',
        'grid_tools',
        'constants',
        'covariance',
        'emulator',
        'model',
        # 'parallel',
        'samplers',
        # 'single',
        'spectrum',
        'utils'
    ])
    def test_first_level(self, module):
        exec('import Starfish.{}'.format(module))

    @pytest.mark.parametrize('module', [
        'grid_tools'
    ])
    def test_second_level(self, module):
        exec('from Starfish.{} import *'.format(module))