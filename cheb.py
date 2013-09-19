import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as Ch
import numpy as np

def test_chebyshev():
    coef = np.array([0.,0.,1.,0.01,0.01])
    coef2 = np.array([0.,0.,1,-1,0])
    myCh = Ch(coef)
    myCh2 = Ch(coef2)
    #xs = np.linspace(0,3.)
    x0 = np.linspace(-1,1)
    #plt.plot(x0, myCh(x0))
    plt.plot(x0, myCh2(x0))
    #plt.plot(xs, myCh2(xs))

    plt.show()


test_chebyshev()

