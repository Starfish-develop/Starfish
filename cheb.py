import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as Ch
import numpy as np
import model as m


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

c = np.array([1,0,0,0])


orders = [21,22]

wls = np.load("GWOri_cf_wls.npy")[orders]
fls = np.load("GWOri_cf_fls.npy")[orders]

sigmas = np.load("sigmas.npy")[orders] #has shape (51, 2299), a sigma array for each order

fmods = m.model(wls,6001,3.5,42,2.1e-28)
#fls = m.model(wls,6020,3.6,40,2e-27)

TT = np.einsum("in,jn->ijn",T,T)

mu = np.array([1,0,0,0])
sigmac = 0.2
D = sigmac**(-2) * np.eye(4)
Dmu = np.einsum("ij,j->j",D,mu)
muDmu = np.einsum("j,j->",mu,Dmu)

a= fmods**2/sigmas**2
A = np.einsum("in,jkn->ijk",a,TT)
#add in prior
A = A + D
detA = np.array(list(map(np.linalg.det, A)))
invA = np.array(list(map(np.linalg.inv, A)))

b = fmods * fls / sigmas**2
B = np.einsum("in,jn->ij",b,T)
B = B + Dmu

g = -0.5 * fls**2/sigmas**2
G = np.einsum("ij->i",g)
G = G - 0.5 * muDmu

#A,B,G are correct

invAB = np.einsum("ijk,ik->ij",invA,B)
BAB = np.einsum("ij,ij->i",B,invAB)

#these are now correct

lnp = 0.5 * np.log((2. * np.pi)**len(orders)/detA) + 0.5 * BAB + G
#print(lnp)
print("Marginalized",np.sum(lnp))

#print(m.lnprob(np.array([6000, 3.5, 42, 0, 2.1e-27, 0.0, 0.0])))

Ac = np.einsum("ijk,k->ij",A,c)
cAc = np.einsum("j,ij->i",c,Ac)
Bc = np.einsum("ij,j->i",B,c)

#to obtain the original, unnormalized P given c
print("Unmarginalized",np.sum(-0.5 * cAc + Bc + G))

#plt.plot(xs,T0)
#plt.plot(xs,T1)
#plt.plot(xs,T2)
#plt.plot(xs,T3)
#plt.show()

