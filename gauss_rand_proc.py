import numpy as np
import matplotlib.pyplot as plt

wls = np.load("data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux.wls.npy")
fls = np.load("data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux.fls.npy")
sigmas = np.load("data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux.sigma.npy")

ind = (wls[22] >= 5140) & (wls[22] <= 5190)
wl = wls[22][ind]
fl = fls[22][ind]
sigma = sigmas[22][ind]
std = np.std(fl)**2
print(std)

xind = (wl >= 5170.9) & (wl <= 5171.3)
wl2 = wl[~xind]
fl2 = fl[~xind]

l = 0.17 #correlation length
nsamples = 10

def func(x0i, x1i, x0v=None, x1v=None):
    return std * np.exp(-0.5 * np.abs((x0v[x0i] -x1v[x1i])/l)**2)

#cov = np.fromfunction(func, (npoints,npoints), xv=np.arange(npoints), yv=np.arange(npoints),dtype=np.int)


#training set
#x = np.linspace(0,10,num=10)
#y = np.cos(x/2.)
#std = np.std(y)**2

x = wl2
y = fl2
#plt.plot(x,y)
#plt.show()

xs = np.arange(5170.5,5171.5,0.05)
#xs = np.linspace(0,5, num = 10)


Kxx = np.fromfunction(func, (len(x),len(x)), x0v=x, x1v=x, dtype=np.int)
Kxxs = np.fromfunction(func, (len(x),len(xs)), x0v=x, x1v=xs, dtype=np.int)
Kxsx = np.fromfunction(func, (len(xs),len(x)), x0v=xs, x1v=x, dtype=np.int)
Kxsxs = np.fromfunction(func, (len(xs),len(xs)), x0v=xs, x1v=xs, dtype=np.int)
#print(Kxx.shape, Kxxs.shape, Kxsx.shape, Kxsxs.shape) 

var = np.average(sigma)**2
Kxx += var * np.eye(len(x))

inv = np.linalg.inv(Kxx)
#print(inv)
cov = Kxsxs - np.dot(Kxsx, np.dot(inv, Kxxs))
mu = np.dot(Kxsx, np.dot(inv, y))
print(mu)
print(cov)

#ys = np.random.multivariate_normal(mu, cov)


ys = [np.random.multivariate_normal(mu, cov) for i in range(nsamples)]

for yi in ys:
    plt.plot(xs,yi)

plt.plot(wl,fl,"b.")
plt.plot(x,y,"bo")
plt.xlim(5170, 5172)
plt.show()


#If we know the covariance function, and a given point is bad, can we predict the values of the nearby data? 
#This might be interesting to sample the lines for each spectrum before we fit
