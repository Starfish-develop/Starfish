from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np
import sys
from emcee.utils import MPIPool

myDataSpectrum = DataSpectrum.open("../data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("../libraries/PHOENIX_submaster.hdf5")

stellar_Starting = {"temp":(6000, 6100), "logg":(3.9, 4.2), "Z":(-0.6, -0.3), "vsini":(4, 6), "vz":(15.0, 16.0), "logOmega":(-19.665, -19.664)}

stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)

#cheb_Starting = {"c1": (-.02, -0.015), "c2": (-.0195, -0.0165), "c3": (-.005, .0)}
cheb_Starting = {"logc0": (-0.02, 0.02), "c1": (-.02, 0.02), "c2": (-0.02, 0.02), "c3": (-.02, 0.02)}
cov_Starting = {"sigAmp":(0.9, 1.1), "logAmp":(-14.4, -14), "l":(0.1, 0.25)}
cov_tuple = C.dictkeys_to_covtuple(cov_Starting)

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cov_tuple=cov_tuple)




def eval0():
    myModel.evaluate()

def eval1():
    myModel.evaluate()

def main():
    print("updating model")
    myModel.update_Model(params)
    myModel.update_Cheb(np.array([0, 0, 0, 0]))
    print("evaluating 0")
    eval0()
    #print("evaluating 1")
    #myModel.update_Cov({"sigAmp": 1, "logAmp":-15 , "l":1})
    #eval1()

if __name__=="__main__":
    main()