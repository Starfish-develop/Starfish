from StellarSpectra.model import Model, SamplerStellarCheb
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface, InterpolationError
import numpy as np

myDataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([21,22,23]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("/home/ian/Grad/Research/Disks/StellarSpectra/libraries/PHOENIX_submaster.hdf5")

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, ("temp", "logg", "Z", "alpha", "vsini", "vz", "Av", "Omega"))

def lnprob_Model(p):
    params = myModel.zip_stellar_p(p)
    try:
        myModel.update_Model(params)
        return myModel.evaluate()
    except InterpolationError:
        return -np.inf

def lnprob_Cheb(p):
    myModel.update_Cheb(p)
    return myModel.evaluate()

def lnprob_Cov(p):
    convert_p_to_dict(p)
    myModel.update_Cov(params)
    return myModel.evaluate()

mySampler = SamplerStellarCheb(lnprob_Model, {"temp":(6200, 6800), "logg":(3.9, 4.2), "Z":(-0.6, -0.1),
                                            "alpha":(0.01, 0.2), "vsini":(0, 10), "vz":(10, 20), "Av":(0,0.1),
                                            "Omega":(2e-20, 4e-20)}, 100)
mySampler.burn_in()
mySampler.run(100)

mySampler.plot()