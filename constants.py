import numpy as np

##################################################
# Constants
##################################################
c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

#n @ 3000: 1.0002915686329712
#n @ 6000: 1.0002769832562917
#n @ 8000: 1.0002750477973053

n_air = 1.000277
c_ang_air = c_ang/n_air
c_kms_air = c_kms/n_air

h = 6.6260755e-27 #erg s

G = 6.67259e-8 #cm3 g-1 s-2
M_sun = 1.99e33 #g
R_sun = 6.955e10 #cm
pc = 3.0856776e18 #cm
AU = 1.4959787066e13 #cm

L_sun = 3.839e33 #erg/s
R_sun = 6.955e10 #cm
F_sun = L_sun / (4 * np.pi * R_sun ** 2) #bolometric flux of the Sun measured at the surface

grid_parameters = frozenset(("temp", "logg", "Z", "alpha")) #Allowed grid parameters
pp_parameters = frozenset(("vsini", "FWHM", "vz", "Av", "Omega")) #Allowed "post processing parameters"
all_parameters = grid_parameters | pp_parameters #the union of grid_parameters and pp_parameters
#Dictionary of allowed variables with default values
var_default = {"temp":5800, "logg":4.5, "Z":0.0, "alpha":0.0, "vsini":0.0, "FWHM": 0.0, "vz":0.0, "Av":0.0, "Omega":1.0}

def dictkeys_to_tuple(mydict):
    if "alpha" in mydict.keys():
        tup = ("temp", 'logg', 'Z', 'alpha')
    else:
        tup = ("temp", 'logg', 'Z')

    if "FWHM" in mydict.keys():
        tup2 = ("vsini", 'FWHM', 'vz', 'Av', 'Omega')
    else:
        tup2 = ("vsini", 'vz', 'Av', 'Omega')

    return tup + tup2


def dict_to_tuple(mydict):
    '''
    Take a parameter dictionary and convert it to a tuple in the standard order.

    :param mydict: input parameter dictionary
    :type mydict: dict
    :returns: sorted tuple which always includes *alpha*
    :rtype: 4-tuple
        '''
    if "alpha" in mydict.keys():
        tup = (mydict["temp"], mydict['logg'], mydict['Z'], mydict['alpha'])
    else:
        tup = (mydict["temp"], mydict['logg'], mydict['Z'], C.var_default['alpha'])

    if "FWHM" in mydict.keys():
        tup2 = (mydict["vsini"], mydict['FWHM'], mydict['vz'], mydict['Av'], mydict['Omega'])
    else:
        tup2 = (mydict["vsini"], mydict['vz'], mydict['Av'], mydict['Omega'])

    return tup + tup2