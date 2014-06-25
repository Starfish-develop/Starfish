#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from StellarSpectra.grid_tools import HDF5Interface
from StellarSpectra.model import Model, ModelHA


'''
Designed to interrogate how instrumental convolution and downsampling the grid may affect the accuracy of the final
generated spectra. Basically, we want to find the minimal resolution such that we are still under some stated
accuracy in parameters (for example 10 K, 0.05 dex in logg and Z).

'''

def get_resid_spec(base, offset):
    '''
    Given a base spectrum and a spectrum slightly offset, return the residual spectrum between the two
    '''
    #Normalize both spectra
    base /= np.mean(base)
    offset /= np.mean(offset)

    #return the absolute "error"
    return np.abs((base - offset)/base)

def get_min_spec(spec_list):
    '''
    Given a list of residual spectra, created with `create_resid_spec', determine for each pixel the minimum offset so
    that we can use this as an error envelope in the approximate spectra.
    '''

    #For each pixel, take the smallest value
    #Vstack the arrays and take the min along axis=1
    arr = np.vstack(spec_list)
    return np.min(arr, axis=0)

class AccuracyComparison:
    '''
    Gather the data products necessary to make a test about accuracy of the reduced grid sizes.

    '''

    def __init__(self, DataSpectrum, Instrument, LibraryHA, LibraryLA, parameters, deltaParameters):
        '''Initialize the comparison object.

        :param DataSpectrum: the spectrum that provides a wl grid + natural resolution
        :type DataSpectrum: :obj:`grid_tools.DataSpectrum`
        :param Instrument: the instrument object on which the DataSpectrum was acquired (ie, TRES, SPEX...)
        :type Instrument: :obj:`grid_tools.Instrument`
        :param LibraryHA: the path to the native resolution spectral library
        :type LibraryHA: string
        :param LibraryLA: the path to the approximate spectral library
        :type LibraryLA: string

        '''

        self.DataSpectrum = DataSpectrum
        self.Instrument = Instrument

        self.HDF5InterfaceHA = HDF5Interface(LibraryHA)
        self.HDF5InterfaceLA = HDF5Interface(LibraryLA)

        print("Bounds of the grids are")
        print("HA", self.HDF5InterfaceHA.bounds)
        print("LA", self.HDF5InterfaceLA.bounds)

        #If the DataSpectrum contains more than one order, we only take the first one. To get behavior with a
        # different order, you should only load that via the DataSpectrum(orders=[22]) flag.
        self.wl = self.DataSpectrum.wls[0]

        self.fullModelLA = Model(self.DataSpectrum, self.Instrument, self.HDF5InterfaceLA, stellar_tuple=("temp",
                    "logg", "Z", "vsini", "vz", "logOmega"), cheb_tuple=("c1", "c2", "c3"), cov_tuple=("sigAmp",
                    "logAmp", "l"), region_tuple=("loga", "mu", "sigma"))
        self.modelLA = self.fullModelLA.OrderModels[0]


        self.fullModelHA = ModelHA(self.DataSpectrum, self.Instrument, self.HDF5InterfaceHA, stellar_tuple=("temp",
                    "logg", "Z", "vsini", "vz", "logOmega"), cheb_tuple=("c1", "c2", "c3"), cov_tuple=("sigAmp",
                    "logAmp", "l"), region_tuple=("loga", "mu", "sigma"))
        self.modelHA = self.fullModelHA.OrderModels[0]

        self.parameters = parameters
        self.deltaParameters = deltaParameters

        self.base = self.get_specHA(self.parameters)
        self.baseLA = self.get_specLA(self.parameters)
        self.approxResid = get_resid_spec(self.base, self.baseLA) #modelHA - modelLA @ parameters

    def get_specHA(self, parameters):
        '''
        Update the model and then query the spectrum

        :param parameters: Dictionary of fundamental stellar parameters
        :type parameters: dict

        :returns: flux spectrum
        '''

        params = parameters.copy()
        params.update({"vsini":0., "vz":0, "logOmega":0.})
        self.fullModelHA.update_Model(params)

        return self.modelHA.get_spectrum()

    def get_specLA(self, parameters):
        '''
        Update the model and then query the spectrum

        :param parameters: Dictionary of fundamental stellar parameters
        :type parameters: dict

        :returns: flux spectrum
        '''

        params = parameters.copy()
        params.update({"vsini":0., "vz":0, "logOmega":0.})
        self.fullModelLA.update_Model(params)

        return self.modelLA.get_spectrum()

    def createEnvelopeSpectrum(self, direction='both'):
        '''
        The parameters should always be specified at a grid point of the HDF5 file.

        For this, do the deltaParameters interpolation.

        Direction specifies whether to do interpolation up (+ 10 K, etc.), down (- 10 K), or
        do both and then find the minimum envelope between the two.
        For now, only up is implemented.

        '''
        #For each key, add the delta parameters
        temp_params = self.parameters.copy()
        temp_params["temp"] += self.deltaParameters["temp"]
        temp_spec = get_resid_spec(self.base, self.get_specHA(temp_params))

        logg_params = self.parameters.copy()
        logg_params["logg"] += self.deltaParameters["logg"]
        logg_spec = get_resid_spec(self.base, self.get_specHA(logg_params))

        Z_params = self.parameters.copy()
        Z_params["Z"] += self.deltaParameters["Z"]
        Z_spec = get_resid_spec(self.base, self.get_specHA(Z_params))

        self.envelope = get_min_spec([temp_spec, logg_spec, Z_spec])

    def plot_quality(self):
        '''
        Visualize the quality of the interpolation.

        Two-panel plot.

        Top: HA and LA spectrum

        Bottom: Residual between HA + LA spectrum and the HA spectrum error bounds for deltaParameters

        '''

        self.createEnvelopeSpectrum()

        print("Spectrum is len ", len(self.wl))
        print("Value at 3134 is ", self.wl[3135], self.base[3135])

        fig, ax = plt.subplots(nrows=2, figsize=(8,6), sharex=True)
        ax[0].plot(self.wl, self.base, "b", label="HA")
        ax[0].plot(self.wl, self.baseLA, "r", label="LA")
        ax[0].legend()
        ax[0].set_ylabel(r"$\propto f_\lambda$")
        ax[0].set_title("Temp={temp:} logg={logg:} Z={Z:}".format(**self.parameters))

        ax[1].semilogy(self.wl, self.approxResid, "k", label="(HA - LA)/HA")
        ax[1].semilogy(self.wl, self.envelope, "b", label="Interp Envelope")
        ax[1].legend()
        ax[1].set_xlabel(r"$\lambda$\AA")
        ax[1].set_ylabel("fractional error")

        return fig


def main():
    from StellarSpectra.spectrum import DataSpectrum
    from StellarSpectra.grid_tools import TRES

    myDataSpectrum = DataSpectrum.open("../../data/WASP14/WASP14-2009-06-14.hdf5", orders=np.array([22]))
    myInstrument = TRES()

    myComp = AccuracyComparison(myDataSpectrum, myInstrument, "../../libraries/PHOENIX_submaster.hdf5",
                                "../../libraries/PHOENIX_objgrid6000.hdf5",
                                {"temp":6000, "logg":4.5, "Z":-0.5}, {"temp":10, "logg":0.05, "Z": 0.05})

    myComp.plot_quality()

if __name__=="__main__":
    main()