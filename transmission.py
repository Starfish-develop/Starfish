import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

__author__ = 'ian'

data = ascii.read("data/TAPAS/WASP14.ipac")
wl = 10 *data['wavelength'] #angstroms
tr = data['transmittance']

plt.plot(wl,tr)
plt.show()
