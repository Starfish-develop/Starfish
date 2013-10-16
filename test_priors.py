import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad

__author__ = 'Ian Czekala'

def pc(c0,sigmac=0.2):
    #mu = sigmac**2
    if c0 > 0:
        return 1/(np.sqrt(np.pi * 2.) * c0 * sigmac) * np.exp(-(np.log(c0))**2/(2 * sigmac**2))
    else:
        return 0.

def pJ(F, c0, fD=10., fM=1., sigma=0.5, sigmac=0.2):
    return 1/(np.sqrt(2. * np.pi) * sigma) * np.exp(-(fD - F * c0 * fM)**2/(2 * sigma**2)) * pc(c0,sigmac)

#pfunc = lambda x: -pc(x[0])
#func = lambda x: -pJ(x[0],x[1])


#print(minimize(pfunc, x0=[1.01]))
##
#print(minimize(func, x0=[10,1.01],method='L-BFGS-B',bounds=((9.5,10.5),(0.9,1.1))))
#print(minimize(func, x0=[9.9,0.9],method='L-BFGS-B',bounds=((9.5,10.5),(0.9,1.1))))

# Questions
# Is it a problem that the prior doesn't actually peak at c0=1? At least it is symmetric.
def p(F):
    '''Integrate out c0'''
    int_func = lambda x: pJ(F, x)
    return quad(int_func,0,np.inf)[0]

fDs = np.random.normal(loc=10.,scale=0.5,size=(200,))
mean = np.mean(fDs)
print(mean)
#plt.hist(fDs)
#plt.show()
fMs = np.ones_like(fDs)
@np.vectorize
def pJ2(F,c0,fD=fDs,fM=fMs,sigma=0.5,sigmac=0.2):
    return 1/(np.sqrt(2. * np.pi) *  c0 * sigmac * (np.sqrt(2 * np.pi) * sigma)**len(fD)) * np.exp(-0.5 * np.sum((fD - F * c0 * fM)**2/sigma**2 + np.log(c0)**2/sigmac**2))


def pG2(F,c0,fD=fDs,fM=fMs,sigma=0.5,sigmac=0.2):
    return 1/((2. * np.pi) *  sigmac * (np.sqrt(2 * np.pi) * sigma)**len(fD)) * np.exp(-0.5 * np.sum((fD - F * c0 * fM)**2/sigma**2 + (c0 - 1)**2/sigmac**2))


# 1) Generate an ensemble of fD values from the Gaussian error distribution
#func = lambda x: -pJ2(x[0],x[1])
#print(minimize(func, x0=[10,1.01],method='L-BFGS-B',bounds=((9.5,10.5),(0.9,1.1))))
#
##print(pJ2(10,1,fDs,fMs))
#FF,cc = np.meshgrid(np.linspace(9.95,10.05),np.linspace(0.95,1.05))
#plt.contour(FF,cc,pJ2(FF,cc))
#plt.plot(10,1,"o")
#plt.show()

@np.vectorize
def p2(F):
    '''Integrate out c0'''
    int_func = lambda x: pJ2(F, x)
    return quad(int_func,0,np.inf,epsabs=1e-100,epsrel=1e-100)[0]

@np.vectorize
def pG(F):
    '''Integrate out c0'''
    int_func = lambda x: pG2(F, x)
    return quad(int_func,0,np.inf,epsabs=1e-100,epsrel=1e-100)[0]

print(pJ2(10.,1))
print(pJ2(10,1.01))

print(p2(10))
print(p2(10.1))

Fs = np.linspace(9.5,10.5,num=100)
cs = np.linspace(0.98,1.02)
#plt.plot(cs, pJ2(10.,cs))
#plt.show()
plt.plot(Fs, p2(Fs))
plt.plot(Fs, pG(Fs))
plt.axvline(mean)
plt.show()