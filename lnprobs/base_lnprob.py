from StellarSpectra.model import Model, StellarSampler, ChebSampler, CovSampler, MegaSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np
import sys
from emcee.utils import MPIPool

myDataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("/home/ian/Grad/Research/Disks/StellarSpectra/libraries/PHOENIX_submaster.hdf5")

stellar_Starting = {"temp":(6000, 6100), "logg":(3.9, 4.2), "Z":(-0.6, -0.3), "vsini":(4, 6), "vz":(12.8, 14), "logOmega":(-19.68, -19.65)}

stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)

cheb_Starting = {"c1": (-.1, 0), "c2": (-.1, .0), "c3": (-.05, .0)}
cov_Starting = {"sigAmp":(0.8, 1.2), "logAmp":(-15, -13), "l":(0.1, 2.0)}
cov_tuple = C.dictkeys_to_covtuple(cov_Starting)

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cov_tuple=cov_tuple)

def lnprob_Model(p):
    params = myModel.zip_stellar_p(p)
    try:
        myModel.update_Model(params)
        return myModel.evaluate()
    except C.ModelError:
        return -np.inf

def lnprob_Cheb(p):
    myModel.update_Cheb(p)
    return myModel.evaluate()

def lnprob_Cov(p):
    params = myModel.zip_Cov_p(p)
    myModel.update_Cov(params)
    return myModel.evaluate()

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

myStellarSampler = StellarSampler(lnprob_Model, stellar_Starting, pool=pool)
myChebSampler = ChebSampler(lnprob_Cheb, cheb_Starting, pool=pool)
myCovSampler = CovSampler(lnprob_Cov, cov_Starting, pool=pool)

myMegaSampler = MegaSampler(samplers=(myStellarSampler, myChebSampler, myCovSampler), burnInCadence=(1, 1, 2), cadence=(10, 10, 10))

myMegaSampler.burn_in(5)
#myMegaSampler.reset()
#myMegaSampler.run(10)
pool.close()
myMegaSampler.plot()
