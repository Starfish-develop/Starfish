"""
Constants important for interacting with spectra
"""
from math import pi

c_ang = 2.99792458e18  # A s^-1
c_kms = 2.99792458e5  # km s^-1

# n @ 3000: 1.0002915686329712
# n @ 6000: 1.0002769832562917
# n @ 8000: 1.0002750477973053

n_air = 1.000277
c_ang_air = c_ang / n_air
c_kms_air = c_kms / n_air

h = 6.6260755e-27  # erg s

G = 6.67259e-8  # cm3 g-1 s-2
M_sun = 1.99e33  # g
R_sun = 6.955e10  # cm
pc = 3.0856776e18  # cm
AU = 1.4959787066e13  # cm

L_sun = 3.839e33  # erg/s
R_sun = 6.955e10  # cm
F_sun = L_sun / (
    4 * pi * R_sun ** 2
)  # bolometric flux of the Sun measured at the surface

# hc / k_B
hc_k = 1.43877735e8  # K AA
