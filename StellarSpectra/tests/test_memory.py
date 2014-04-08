import numpy as np
from scipy.interpolate import interp1d
import h5py
import yaml
import gc
import pyfftw
import emcee
import bz2
from astropy.io import ascii,fits
from scipy.integrate import trapz

@profile
def base():
    print("base")

@profile
def base2():
    print("base2")

@profile
def base3():
    print("base3")


base()

@profile
def loadnp():
    a = np.load("test_spec.npy")
    return a

wl,fl = loadnp()

@profile
def integrate():
    x = np.linspace(0,20,num=1000000)
    ans = trapz(x,x)
    return ans

@profile
def integrate2():
    ans = trapz(fl/2., wl/3.)
    return ans

@profile
def integrate3():
    x = np.linspace(0,20,num=1000000)
    ans = trapz(x,x)
    return ans

base2()

ans = integrate()

ans2 = integrate2()

ans3 = integrate3()

base3()

#There will always be about a base 10MB penalty to any Python interpreter.

#No modules = 10MB
#+numpy as np = 18MB
#+scipy = 25MB
#+h5py = 33MB
#+yaml = 34MB
#+gc = 34MB
#+pyfftw = 37MB
#+emcee = 37MB
#+bz2 = 37MB
#+astropy = 55MB

#trapz is an interesting method. Once the integrator is defined, it keeps it's memory. Subsequent calls to this
#might not increase the memory.
