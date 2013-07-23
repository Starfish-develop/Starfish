import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import *

#xc = np.array([-2,-1,0,1,2])
c = np.array([1/16.,1/4.,3/8.,1/4.,1/16.])

@np.vectorize
def gauss(x, sigma=1.):
    return 1./(np.sqrt(2. * np.pi) * sigma) * np.exp(-0.5 * x**2/sigma**2)

xs = np.linspace(-10,10)
plt.plot(xc,c,"o")
plt.plot(xs,gauss(xs))
plt.plot(xs,1/2*gauss(xs,sigma=2))
plt.plot(xs,gauss(xs) - 1/2*gauss(xs, sigma=2))
plt.show()

#sc = fftshift(c)
#fsc = fft(sc)
#ifsc = fftshift(fsc)
#plt.plot(ifsc)
#plt.show()
#
