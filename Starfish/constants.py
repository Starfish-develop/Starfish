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


grid_parameters = ("temp", "logg", "Z", "alpha") #Allowed grid parameters
grid_set = frozenset(grid_parameters)

pp_parameters = ("vsini", "FWHM", "vz", "Av", "logOmega", "vz1", "Av1", "logOmega1") #Allowed "post processing
# parameters"
pp_set = frozenset(pp_parameters)

stellar_parameters = grid_parameters + pp_parameters
stellar_set = grid_set | pp_set #the union of grid_parameters and pp_parameters


#Dictionary of allowed variables with default values
var_default = {"temp":5800, "logg":4.5, "Z":0.0, "alpha":0.0, "vsini":0.0, "FWHM": 0.0, "vz":0.0, "Av":0.0, "logOmega":0.0}

cov_global_parameters = ("sigAmp", "logAmp", "l")
cov_region_parameters = ("logAmp", "mu", "sigma")

def dictkeys_to_tuple(keys):
    '''
    Helper function to convert a dictionary of starting stellar parameters to a tuple list

    :param keys: keys to sort into a tuple
    :type keys: dict.keys() view
    '''

    tup = ()
    for param in stellar_parameters:
        #check if param is in keys, if so, add to the tuple
        if param in keys:
            tup += (param,)

    return tup

def dictkeys_to_cov_global_tuple(keys):
    '''
    Helper function to convert a dictionary of starting stellar parameters to a tuple list

    :param keys: keys to sort into a tuple
    :type keys: dict.keys() view
    '''

    tup = ()
    for param in cov_global_parameters:
        #check if param is in keys, if so, add to the tuple
        if param in keys:
            tup += (param,)

    return tup

def dictkeys_to_cov_region_tuple(keys):
    '''
    Helper function to convert a dictionary of starting stellar parameters to a tuple list

    :param keys: keys to sort into a tuple
    :type keys: dict.keys() view
    '''

    tup = ()
    for param in cov_region_parameters:
        #check if param is in keys, if so, add to the tuple
        if param in keys:
            tup += (param,)

    return tup

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
        tup = (mydict["temp"], mydict['logg'], mydict['Z'], var_default['alpha'])

    if "FWHM" in mydict.keys():
        tup2 = (mydict["vsini"], mydict['FWHM'], mydict['vz'], mydict['Av'], mydict['Omega'])
    else:
        tup2 = (mydict["vsini"], mydict['vz'], mydict['Av'], mydict['Omega'])

    return tup + tup2


class ModelError(Exception):
    '''
    Raised when Model parameters cannot be found in the grid.
    '''
    def __init__(self, msg):
        self.msg = msg

class GridError(Exception):
    '''
    Raised when a spectrum cannot be found in the grid.
    '''
    def __init__(self, msg):
        self.msg = msg

class InterpolationError(Exception):
    '''
    Raised when the :obj:`Interpolator` or :obj:`IndexInterpolator` cannot properly interpolate a spectrum,
    usually grid bounds.
    '''
    def __init__(self, msg):
        self.msg = msg

class RegionError(Exception):
    '''
    Raised when a RegionCovarianceMatrix cannot be instantiated.
    '''
    def __init__(self, msg):
        self.msg = msg


def get_git_commit():
    '''
    Return the git branch and commit as a string
    '''

    #Figure out what the relative path is to base
    import Starfish, subprocess
    base = Starfish.__file__[:-21]
    gitbase = base + "/.git"

    proc = subprocess.Popen(["git", "--git-dir={}".format(gitbase), "--work-tree={}".format(base), "log", "-1",
                            "--date=short", "--oneline"], stdout=subprocess.PIPE)
    out, err = proc.communicate()
    return out.decode('UTF-8')
