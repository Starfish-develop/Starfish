import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from scipy.ndimage.filters import convolve
import asciitable
import pyfits as pf

h0 = np.array([1/16.,1/4.,3/8.,1/4.,1/16.])


@np.vectorize
def gauss(x, mu=0, sigma=1.):
    return 1./(np.sqrt(2. * np.pi) * sigma) * np.exp(-0.5 * (x-mu)**2/sigma**2)

def write_data():
    wls = np.linspace(50,300.,num=1000)
    flux = (wls)**(0.2) + 10*gauss(wls, mu=150, sigma=2) - 3.*gauss(wls, mu=240, sigma=1) + normal(scale=0.1,size=(len(wls,)))
    asciitable.write({'w':wls,'f':flux},"data.txt",names=['w','f'])

#write_data()

#data = asciitable.read('data.txt')
#wls = data['w']
#flux = data['f']


flux_file = pf.open("../HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
#flux_file = pf.open("../lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
wl_file = pf.open("../WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

flux = flux_file[0].data
wl = wl_file[0].data
ind = (wl > 3000) & (wl < 10000)

wls = wl[ind]
flux = flux[ind]


global i
j = 1.

def double_filter(h):
    new_shape = 2. * len(h) - 1
    print(new_shape)
    hi = np.zeros((new_shape,))
    global j
    stride = 2**j
    hi[::stride] = h0
    j += 1
    return hi

iterations = 16
h = [h0]
c = np.zeros((iterations,len(flux)))
ws = np.zeros((iterations,len(flux)))
c[0,:] = flux

for i in range(1,iterations):
    h.append(double_filter(h[i-1]))
    c[i,:] = convolve(c[i-1,:],h[i-1],mode="nearest")
    ws[i,:] = c[i,:] - c[i-1,:]


def plot_cs():
    fig, ax = plt.subplots(nrows=iterations,ncols=1,sharex=True,sharey=True,figsize=(8,12))
    for i in range(iterations):
        ax[i].plot(wls,c[i,:])
    fig.subplots_adjust(top=0.98,bottom=0.05,right=0.98,left=0.1,hspace=0.25)
    plt.savefig("PHOENIX_cont4000K.png")

def main():
    plot_cs()

if __name__=="__main__":
    main()
