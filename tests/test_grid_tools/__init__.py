import os
from itertools import product

import pytest

from Starfish.grid_tools import download_PHOENIX_models, Instrument

@pytest.fixture(scope='session')
def PHOENIXModels():
    params = product(
        (6000, 6100, 6200),
        (4.0, 4.5, 5.0),
        (0.0, -0.5, -1.0)
    )
    test_base = os.path.dirname(os.path.dirname(__file__))
    outdir = os.path.join(test_base, 'data', 'phoenix')
    download_PHOENIX_models(params, outdir)
    yield outdir


@pytest.fixture(scope='session')
def AlphaPHOENIXModels():
    params = product(
        (6100,),
        (4.5,),
        (0.0,),
        (-0.2,)
    )
    test_base = os.path.dirname(os.path.dirname(__file__))
    outdir = os.path.join(test_base, 'data', 'phoenix')
    download_PHOENIX_models(params, outdir)
    yield outdir

@pytest.fixture
def mock_instrument():
    yield Instrument('Test Instrument', FWHM=45.0, wl_range=(5e3, 6e3))