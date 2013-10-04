import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as Ch
import numpy as np


def test_chebyshev():
    '''Domain controls the x-range, while window controls the y-range.'''
    coef = np.array([0., 1.])
    #coef2 = np.array([0.,0.,1,-1,0])
    myCh = Ch(coef, window=[-10, 10])
    #Domain c
    #myCh2 = Ch(coef2)
    #xs = np.linspace(0,3.)
    x0 = np.linspace(-1, 1)
    plt.plot(x0, myCh(x0))
    #plt.plot(x0, myCh2(x0))
    #plt.plot(xs, myCh2(xs))

    plt.show()


#test_chebyshev()


xs = np.arange(2299)
T0 = np.ones_like(xs)

Ch1 = Ch([0,1], domain=[0,2298])
T1 = Ch1(xs)

Ch2 = Ch([0,0,1],domain=[0,2298])
T2 = Ch2(xs)

Ch3 = Ch([0,0,0,1],domain=[0,2298])
T3 = Ch3(xs)

T = np.array([T0,T1,T2,T3]) #multiply this by the flux and sigma vector for each order
TT = np.einsum("in,jn->ijn",T,T)

wls = np.load("GWOri_cf_wls.npy")[22]
fls = np.load("GWOri_cf_fls.npy")[22]

sigmas = np.load("sigmas.npy")[22] #has shape (51, 2299), a sigma array for each order

all = TT*wls*fls
A = np.sum(all,axis=-1)

#plt.plot(xs,T0)
#plt.plot(xs,T1)
#plt.plot(xs,T2)
#plt.plot(xs,T3)
#plt.show()

