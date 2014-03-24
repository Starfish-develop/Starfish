import numpy as np

from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface

myDataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("/home/ian/Grad/Research/Disks/StellarSpectra/libraries/PHOENIX_submaster.hdf5")

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=("temp", "logg", "Z", "vsini", "vz", "logOmega"), 
                cov_tuple=("sigAmp", "logAmp", "l"))

params = {"temp":6200, "logg":4.0, "Z":-0.2, "vsini":10, "vz":15, "logOmega":-22}



def eval0():
    myModel.evaluate()

def eval1():
    myModel.evaluate()

def main():
    print("updating model")
    myModel.update_Model(params)
    myModel.update_Cheb(np.array([0, 0, 0]))
    print("evaluating 0")
    eval0()
    #print("evaluating 1")
    #myModel.update_Cov({"sigAmp": 1, "logAmp":-15 , "l":1})
    #eval1()

if __name__=="__main__":
    main()