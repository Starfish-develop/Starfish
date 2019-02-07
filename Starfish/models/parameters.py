import os
import json
from itertools import zip_longest

import numpy as np

from Starfish import config


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class SpectrumParameter:
    '''
    An object holding the collection of parameters shared between all orders.

    :param grid: parameters corresponding to the dimensions of the grid.
    :type grid: 1D np.array
    '''

    def __init__(self, grid_params, vz=None, vsini=None, logOmega=None, Av=None, cheb=(None,)):
        self.grid_params = np.array(grid_params)
        self.vz = vz
        self.vsini = vsini
        self.logOmega = logOmega  # log10Omega
        self.Av = Av
        # Fix the linear Chebyshev component to avoid degeneracy with Omega
        self.cheb = np.array([1] + list(cheb))

    def to_array(self):
        array = self.grid_params.tolist() + [self.vz, self.vsini, self.logOmega, self.Av] + self.cheb[1:].tolist()
        return np.array(array)

    @classmethod
    def from_array(cls, array, ngrid):
        grid_params = array[:ngrid]
        vz, vsini, logOmega, Av = array[ngrid:ngrid + 4]
        c = array[ngrid + 4:]
        return cls(grid_params, vz, vsini, logOmega, Av, c)

    def save(self, filename):
        '''
        Save the parameters to a JSON file
        '''
        with open(filename, 'w') as f:
            json.dump(self, f, cls=SpectrumParameterEncoder, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d):
        '''
        Load the parameters from a dictionary, e.g., from the config file

        :param d: dictionary of parameters
        :type d: dictionary
        '''
        d['grid_params'] = np.array(d['grid_params'])
        d['cheb'] = np.array(d['cheb'])
        return cls(**d)

    @classmethod
    def load(cls, filename):
        '''
        Load the parameters from a JSON file
        '''
        with open(filename) as f:
            read = json.load(f)  # read is a dictionary

        return cls.from_dict(read)

    def __repr__(self):
        return "grid_params:{} vz:{} vsini:{} logOmega:{} Av:{} Cheb:{}".format(self.grid_params, self.vz, self.vsini,
                                                                             self.logOmega, self.Av, self.cheb)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        equal = True
        equal &= np.allclose(self.grid_params, other.grid_params)
        equal &= self.vz == other.vz
        equal &= self.vsini == other.vsini
        equal &= self.logOmega == other.logOmega
        equal &= self.Av == other.Av
        try:
            equal &= np.allclose(self.cheb, other.cheb, equal_nan=True)
        except TypeError:
            equal &= self.cheb == other.cheb
        return equal



class SpectrumParameterEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            mydict = {
                "grid_params": o.grid_params.tolist(),
                "vz": o.vz,
                "vsini": o.vsini,
                "logOmega": o.logOmega,
                "Av": o.Av,
                "cheb": o.cheb[1:].tolist()
            }
        except TypeError:
            pass
        else:
            return mydict
        # Let the base class default method raise the TypeError, if there is one
        return json.JSONEncoder.default(self, o)


class EchelleParameter:
    pass


class EchelleParameterEncoder(json.JSONEncoder):
    pass


class PhiParam:
    '''
    An object holding the collection of parameters specific to a single order.
    '''

    def __init__(self, spectrum_id, order, fix_c0=False, cheb=np.zeros((config["cheb_degree"],)),
                 sigAmp=config["Phi"]["sigAmp"], logAmp=config["Phi"]["logAmp"],
                 l=config["Phi"]["l"], regions=None):
        self.spectrum_id = spectrum_id
        self.order = order
        self.fix_c0 = fix_c0
        self.cheb = cheb
        self.sigAmp = sigAmp
        self.logAmp = logAmp
        self.l = l
        self.regions = regions

    def toarray(self):
        '''
        Return parameters formatted as a numpy array.
        '''
        p = list(self.cheb) + [self.sigAmp, self.logAmp, self.l]
        if self.regions is not None:
            p += self.regions.flatten().tolist()

        return np.array(p)

    def save(self, fname="phi.json"):
        dirname = os.path.dirname(fname)
        name = config.specfmt.format(self.spectrum_id, self.order) + os.path.basename(fname)
        outname = os.path.join(dirname, name)
        with open(outname, 'w') as f:
            json.dump(self, f, cls=PhiEncoder, indent=2, sort_keys=True)
        return outname

    @classmethod
    def load(cls, fname):
        '''
        Load the parameters from a JSON file
        '''
        with open(fname, "r") as f:
            read = json.load(f)  # read is a dictionary

        read["cheb"] = np.array(read["cheb"])

        # Try to read regions
        if "regions" in read:
            read["regions"] = np.array(read["regions"])
        else:
            read["regions"] = None
        return cls(**read)

    def __repr__(self):
        return "spectrum_id:{} order:{} fix_c0:{} cheb:{} sigAmp:{} logAmp:{} l:{} regions:{}".format(self.spectrum_id,
                                                                                                      self.order,
                                                                                                      self.fix_c0,
                                                                                                      self.cheb,
                                                                                                      self.sigAmp,
                                                                                                      self.logAmp,
                                                                                                      self.l,
                                                                                                      self.regions)


class PhiEncoder(json.JSONEncoder):
    '''
    Serialize an instance of o=PhiParam() to JSON
    '''

    def default(self, o):
        try:
            mydict = {"spectrum_id": o.spectrum_id,
                      "order": o.order,
                      "fix_c0": o.fix_c0,
                      "cheb": o.cheb.tolist(),
                      "sigAmp": o.sigAmp,
                      "logAmp": o.logAmp,
                      "l": o.l}
            if o.regions is not None:
                mydict["regions"] = o.regions.tolist()
        except TypeError:
            pass
        else:
            return mydict
        # Let the base class default method raise the TypeError, if there is one
        return json.JSONEncoder.default(self, o)
