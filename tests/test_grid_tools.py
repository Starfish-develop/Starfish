import logging
from itertools import product
from urllib.request import urlretrieve
import socket

import pytest

from Starfish.grid_tools import *

log = logging.getLogger(__file__)


@pytest.fixture(scope='session')
def tmpPHOENIXModels(tmpdir_factory):
    params = product(
        (6000, 6100, 6200),
        (4.0, 4.5, 5.0),
        (0.0, 0.5, 1.0)  # Note these are actually negative but I am going to hard-code the minus sign
    )
    outdir = os.path.join('data', 'phoenix')
    wave_url = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    wave_file = os.path.join(outdir, 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    flux_file_formatter = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{2:02.1f}' \
                          '/lte{0:05.0f}-{1:03.2f}-{2:02.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    output_formatter = 'Z-{2:02.1f}/lte{0:05.0f}-{1:03.2f}-{2:02.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

    os.makedirs(outdir, exist_ok=True)
    # Download step
    log.info('Starting Download of PHOENIX ACES models')
    socket.setdefaulttimeout(120)
    if not os.path.exists(wave_file):
        urlretrieve(wave_url, wave_file)
    for p in params:
        url = flux_file_formatter.format(*p)
        output_file = os.path.join(outdir, output_formatter.format(*p))
        if not os.path.exists(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            urlretrieve(url, output_file)

    yield outdir


class TestRawGridInterface:

    @pytest.fixture(scope='class')
    def rawgrid(self):
        params = {
            "temp": (6000, 6100, 6200),
            "logg": (4.0, 4.5, 5.0),
            "Z"   : (-0.5, 0.0)
        }
        yield RawGridInterface("PHOENIX",
                               list(params),
                               params.values(),
                               wl_range=(3000, 64000))

    @pytest.mark.skip("There is no implementation of this in the code, but could be useful. At the super-class level "
                      "it's tough to be able to check all params for sensible keys")
    def test_initialize(self):
        params = {
            "temp": (6000, 6100, 6200),
            "logg": (4.0, 4.5, 5.0),
            "Z"   : (-0.5, 0.0),
            "bunnies": ("furry", "happy"),
        }
        with pytest.raises(KeyError) as e:
            RawGridInterface("PHOENIX",
                             list(params),
                             params.values(),
                             wl_range=(3000, 54000))

    def test_check_params(self, rawgrid):
        rawgrid.check_params((6100, 4.5, 0.0))

    def test_check_params_extra(self, rawgrid):
        with pytest.raises(ValueError) as e:
            rawgrid.check_params((6100, 4.5, 0.0, 'John Cena'))

    def test_check_params_out_of_bounds(self, rawgrid):
        with pytest.raises(ValueError) as e:
            rawgrid.check_params((6100, 4.5, 0.5))

    def test_implementation_error(self, rawgrid):
        with pytest.raises(NotImplementedError):
            rawgrid.load_flux((6100, 4.5, 0.0))


class TestPHOENIXGridInterface:

    @pytest.fixture(scope='class')
    def grid(self, tmpPHOENIXModels):
        yield PHOENIXGridInterface(base=tmpPHOENIXModels)

    @pytest.mark.skip("No alpha phoenix are downloaded, and I don't want to download them yet")
    def test_check_params_alpha(self, grid):
        assert grid.check_params((6100, 4.5, 0.0, 0.2))


    def test_load_flux(self, grid):
        fl, header = grid.load_flux((6100, 4.5, 0.0, 0.0))
        assert len(fl) == 1540041
        assert header['PHXTEFF'] == 6100
        assert header['PHXLOGG'] == 4.5
        assert header['PHXM_H'] == 0.0
        assert header['PHXALPHA'] == 0.0

    @pytest.mark.skip("No alpha phoenix are downloaded, and I don't want to download them yet")
    def test_load_alpha(self, grid):
        grid.load_flux((6100, 4.5, 0.0, 0.2))


    def test_load_flux_metadata(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5, 0.0, 0.0))
        assert isinstance(hdr, dict)

    def test_bad_base(self):
        # Set a different base location, should raise an error because on this machine there is not file.
        with pytest.raises(ValueError) as e:
            PHOENIXGridInterface(base="wrong_base/")


    def test_no_air(self, tmpPHOENIXModels):
        grid = PHOENIXGridInterface(air=False, base=tmpPHOENIXModels)
        fl, hdr = grid.load_flux((6100, 4.5, 0.0, 0.0))
        assert hdr['air'] == False


    def test_no_norm(self, grid):
        fl, hdr = grid.load_flux((6100, 4.5, 0.0, 0.0), norm=False)
        assert hdr['norm'] == False
