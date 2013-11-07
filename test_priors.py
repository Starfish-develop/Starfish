import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

__author__ = 'Ian Czekala'

# Questions

# Let fD = 10
# fM = 1

fDs = np.random.normal(loc=10.,scale=2.,size=(400,))
mean = np.mean(fDs)
print(mean)

fMs = np.ones_like(fDs)

@np.vectorize
def pJ3(F,c0,fD=fDs,fM=fMs,sigma=2.,sigmac=1):
    return 1e299/(np.sqrt(2. * np.pi) *  c0 * sigmac * (np.sqrt(2 * np.pi) * sigma)**len(fD)) * np.exp(-0.5 * np.sum((fD - F * c0 * fM)**2/sigma**2) -0.5 * np.log(c0)**2/sigmac**2)


def pG3(F,c0,fD=fDs,fM=fMs,sigma=2.,sigmac=1):
    return 1e299/((2. * np.pi) *  sigmac * (np.sqrt(2 * np.pi) * sigma)**len(fD)) * np.exp(-0.5 * np.sum((fD - F * c0 * fM)**2/sigma**2) -0.5 * (c0 - 1)**2/sigmac**2)


@np.vectorize
def p2(F,sigmac=1):
    '''Integrate out c0, what is the inference on the flux-factor?'''
    int_func = lambda x: pJ3(F, x, sigmac=sigmac)
    return quad(int_func,0,np.inf,epsabs=1e-100,epsrel=1e-100)[0]

@np.vectorize
def pG(F,sigmac=1):
    '''Integrate out c0, what is the inference on the flux-factor?'''
    int_func = lambda x: pG3(F, x, sigmac=sigmac)
    return quad(int_func,0,np.inf,epsabs=1e-100,epsrel=1e-100)[0]

def compare_p2_pG():
    Fs = np.linspace(0,30,num=400)
    plt.axvline(mean,color="k")
    plt.plot(Fs, p2(Fs,sigmac=0.2),"b-", label='LN 0.2')
    plt.plot(Fs, pG(Fs,sigmac=0.2),"g-", label='N 0.2')
    plt.plot(Fs, p2(Fs,sigmac=0.6),"b:", label='LN 0.6')
    plt.plot(Fs, pG(Fs,sigmac=0.6),"g:", label='G 0.6')
    plt.legend(loc='upper left')
    plt.show()

compare_p2_pG()


#Testing for combined mixture on 11/6/13
#def pc0(c0, sigmac0 = 0.1):
#    return 1. / (np.sqrt(2 * np.pi) * sigmac0 * c0) * np.exp(- np.log(c0)**2/(2 * sigmac0**2))
#
#

# Make plots at sigma = 0.3, 0.4, 0.5....
#c0s = np.random.lognormal(mean=0, sigma=0.4, size=10000)
#c1s = np.random.normal(loc=0, scale=0.2, size=10000)
#c1p = c0s * c1s
#
#fig,ax = plt.subplots(nrows=3)
#ax[0].hist(c0s,bins=100, range=(0,2))
#ax[1].hist(c1s,bins=40, range=(-0.5, 0.5))
#ax[2].hist(c1p,bins=40, range=(-0.2,0.2))
#ax[2].hist(c1p,bins=40, range=(-0.5, 0.5))
#plt.show()
#
