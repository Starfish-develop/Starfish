import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as Ch
import numpy as np

def test_chebyshev():
    '''Domain controls the x-range, while window controls the y-range.'''
    coef = np.array([0.,1.])
    #coef2 = np.array([0.,0.,1,-1,0])
    myCh = Ch(coef,window=[-10,10])
    #Domain c
    #myCh2 = Ch(coef2)
    #xs = np.linspace(0,3.)
    x0 = np.linspace(-1,1)
    plt.plot(x0, myCh(x0))
    #plt.plot(x0, myCh2(x0))
    #plt.plot(xs, myCh2(xs))

    plt.show()


test_chebyshev()

