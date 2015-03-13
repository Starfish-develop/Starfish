import numpy as np
import Starfish
from . import constants as C
from .grid_tools import Interpolator
import json
import h5py
import logging
import matplotlib.pyplot as plt
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class ThetaParam:
    '''
    An object holding the collection of parameters shared between all orders.

    :param grid: parameters corresponding to the dimensions of the grid.
    :type grid: 1D np.array
    '''
    def __init__(self, grid, vz=0.0, vsini=0.0, logOmega=0.0, Av=0.0):
        self.grid = grid
        self.vz = vz
        self.vsini = vsini
        self.logOmega = logOmega #log10Omega
        self.Av = Av

    def save(self, fname="theta.json"):
        '''
        Save the parameters to a JSON file
        '''
        f = open(fname, 'w')
        json.dump(self, f, cls=ThetaEncoder, indent=2, sort_keys=True)
        f.close()

    @classmethod
    def from_dict(cls, d):
        '''
        Load the parameters from a dictionary, e.g., from the config file

        :param d: dictionary of parameters
        :type d: dictionary
        '''
        d["grid"] = np.array(d["grid"])
        return cls(**d)

    @classmethod
    def load(cls, fname="theta.json"):
        '''
        Load the parameters from a JSON file
        '''
        f = open(fname, "r")
        read = json.load(f) # read is a dictionary
        f.close()
        read["grid"] = np.array(read["grid"])
        return cls(**read)

    def __repr__(self):
        return "grid:{} vz:{} vsini:{} logOmega:{} Av:{}".format(self.grid, self.vz, self.vsini, self.logOmega, self.Av)

class ThetaEncoder(json.JSONEncoder):
    '''
    Serialize an instance of o=ThetaParam() to JSON
    '''
    def default(self, o):
        try:
            mydict = {"grid":o.grid.tolist(),
                "vz":o.vz,
                "vsini":o.vsini,
                "logOmega":o.logOmega,
                "Av":o.Av}
        except TypeError:
            pass
        else:
            return mydict
        # Let the base class default method raise the TypeError, if there is one
        return json.JSONEncoder.default(self, o)

class PhiParam:
    '''
    An object holding the collection of parameters specific to a single order.
    '''
    def __init__(self, spectrum_id, order, fix_c0=False, cheb=np.zeros((Starfish.config["cheb_degree"],)),
        sigAmp=1.0, logAmp=0.0, l=10.0, regions=None):
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
        p = self.cheb.tolist() + [self.sigAmp, self.logAmp, self.l]
        if self.regions is not None:
            p += self.regions.flatten().tolist()

        return np.array(p)

    def save(self, fname="phi.json"):
        f = open(Starfish.specfmt.format(self.spectrum_id, self.order) + fname, 'w')
        json.dump(self, f, cls=PhiEncoder, indent=2, sort_keys=True)
        f.close()

    @classmethod
    def load(cls, fname):
        '''
        Load the parameters from a JSON file
        '''
        f = open(fname, "r")
        read = json.load(f) # read is a dictionary
        f.close()
        read["cheb"] = np.array(read["cheb"])

        # Try to read regions
        if "regions" in read:
            read["regions"] = np.array(read["regions"])
        else:
            read["regions"] = None
        return cls(**read)

    def __repr__(self):
        return "spectrum_id:{} order:{} fix_c0:{} cheb:{} sigAmp:{} logAmp:{} l:{} regions:{}".format(self.spectrum_id, self.order, self.fix_c0, self.cheb, self.sigAmp, self.logAmp, self.l, self.regions)


class PhiEncoder(json.JSONEncoder):
    '''
    Serialize an instance of o=PhiParam() to JSON
    '''
    def default(self, o):
        try:
            print("Calling O:", o)
            mydict = {"spectrum_id":o.spectrum_id,
                "order": o.order,
                "fix_c0": o.fix_c0,
                "cheb": o.cheb.tolist(),
                "sigAmp":o.sigAmp,
                "logAmp":o.logAmp,
                "l":o.l}
            if o.regions is not None:
                mydict["regions"] = o.regions.tolist()
        except TypeError:
            pass
        else:
            return mydict
        # Let the base class default method raise the TypeError, if there is one
        return json.JSONEncoder.default(self, o)
