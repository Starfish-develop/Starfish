from dataclasses import dataclass
from typing import Tuple

# TODO convert to dataclass

# Convert R to FWHM in km/s by \Delta v = c/R
@dataclass
class Instrument:
    """
    Object describing an instrument. This will be used by other methods for
    processing raw synthetic spectra.

    Parameters
    ----------
    name: string
        name of the instrument
    FWHM: float
        the FWHM of the instrumental profile in km/s
    wl_range: Tuple (low, high)
        wavelength range of instrument
    oversampling: float, optional
        how many samples fit across the :attr:`FWHM`. Default is 4.0
    """

    name: str
    FWHM: float
    wl_range: Tuple[float]
    oversampling: float = 4.0

    def __str__(self):
        """
        Prints the relevant properties of the instrument.
        """
        return (
            "instrument Name: {}, FWHM: {:.1f}, oversampling: {:.0f}, "
            "wl_range: {}".format(
                self.name, self.FWHM, self.oversampling, self.wl_range
            )
        )


class TRES(Instrument):
    """TRES instrument"""

    def __init__(self, name="TRES", FWHM=6.8, wl_range=(3500, 9500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        # sets the FWHM and wl_range


class Reticon(Instrument):
    """Reticon instrument"""

    def __init__(self, name="Reticon", FWHM=8.5, wl_range=(5145, 5250)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class KPNO(Instrument):
    """KNPO instrument"""

    def __init__(self, name="KPNO", FWHM=14.4, wl_range=(6250, 6650)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class SPEX(Instrument):
    """SPEX instrument at IRTF in Hawaii"""

    def __init__(self, name="SPEX", FWHM=150.0, wl_range=(7500, 54000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class SPEX_SXD(SPEX):
    """SPEX instrument at IRTF in Hawaii short mode (reduced wavelength range)"""

    def __init__(self, name="SPEX_SXD"):
        super().__init__(name=name, wl_range=(7500, 26000))


class SPEX_PRISM(SPEX):
    """SPEX instrument at IRTF in Hawaii low-res PRISM mode (reduced wavelength range)"""

    def __init__(self, name="SPEX_SXD"):
        super().__init__(name=name, FWHM=1500, wl_range=(7500, 26000))


class IGRINS(Instrument):
    """IGRINS Instruments Abstract Class"""

    def __init__(self, wl_range, name="IGRINS"):
        super().__init__(name=name, FWHM=7.5, wl_range=wl_range)
        self.air = False


class IGRINS_H(IGRINS):
    """IGRINS H band instrument"""

    def __init__(self, name="IGRINS_H", wl_range=(14250, 18400)):
        super().__init__(name=name, wl_range=wl_range)


class IGRINS_K(IGRINS):
    """IGRINS K band instrument"""

    def __init__(self, name="IGRINS_K", wl_range=(18500, 25200)):
        super().__init__(name=name, wl_range=wl_range)


class ESPaDOnS(Instrument):
    """ESPaDOnS instrument"""

    def __init__(self, name="ESPaDOnS", FWHM=4.4, wl_range=(3700, 10500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class DCT_DeVeny(Instrument):
    """DCT DeVeny spectrograph instrument."""

    def __init__(self, name="DCT_DeVeny", FWHM=105.2, wl_range=(6000, 10000)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)


class WIYN_Hydra(Instrument):
    """WIYN Hydra spectrograph instrument."""

    def __init__(self, name="WIYN_Hydra", FWHM=300.0, wl_range=(5500, 10500)):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
