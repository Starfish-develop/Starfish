#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev as Ch
from echelle_io import rechellenpflat,load_masks
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import FormatStrFormatter as FSF
import acor

subdir = "order22/"
#subdir = ""

chain = np.load("output/" + subdir + "chain.npy")
nwalkers = chain.shape[0]
nsteps = chain.shape[1]
#Give flatchain, too
s = chain.shape
flatchain = chain.reshape(s[0] * s[1], s[2])
#flatchain = flatchain[80000:]
lnchain = np.load("output/" + subdir + "lnprobchain.npy")
lnflatchain = lnchain.flatten()
#lnchain = lnchain[80000:]
nparams = flatchain.shape[1]

#Load normalized order spectrum
#wls, fls = rechellenpflat("GWOri_cf")

def plot_2d_all():
    A_grid = np.load("A.npy")
    B_grid = np.load("B.npy")
    L = np.load("L.npy")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contour(A_grid,B_grid,L)
    ax.clabel(CS, inline=1, fontsize=10,fmt = '%2.0f')
    ax.plot(dataa['a'],dataa['b'],'b',lw=0.2)
    ax.plot(dataa['a'],dataa['b'],"bo",label="A")
    ax.plot(datab['a'],datab['b'],'g',lw=0.2)
    ax.plot(datab['a'],datab['b'],"go",label="B")
    ax.plot(datac['a'],datac['b'],'r',lw=0.2)
    ax.plot(datac['a'],datac['b'],"ro",label="C")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.legend(loc="lower right")
    fig.savefig("2d_all.eps")

def calc_mean_and_sigma(data):
    burn_in = 10000
    mean_m = np.mean(data["m"][burn_in:])
    mean_b = np.mean(data["b"][burn_in:])
    mean_P = np.mean(data["Pb"][burn_in:])
    mean_Y = np.mean(data["Yb"][burn_in:])
    mean_V = np.mean(data["Vb"][burn_in:])
    sigma_m = np.std(data["m"][burn_in:])
    sigma_b = np.std(data["b"][burn_in:])

    print("m = %.3f +- %.3f" % (mean_m, sigma_m))
    print("b = %.3f +- %.3f" % (mean_b, sigma_b))
    print(mean_P,mean_Y,mean_V)
    print("Acceptance Ratio %.2f" % (len(np.where(data['accept'][burn_in:] == 'True')[0])/len(data['accept'][burn_in:])),)
    return (mean_m,mean_b)


T_points = np.array([2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,6300,6400,6500,6600,6700,6800,6900,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000,9200,9400,9600,9800,10000,10200,10400,10600,10800,11000,11200,11400,11600,11800,12000])
#for our case
Ts = T_points[17:47]
Tbin_edges = np.diff(Ts)/2 + Ts[:-1]
loggbin_edges = [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.0]

def hist_param(flatchain):
    fig, axes = plt.subplots(nrows=nparams,ncols=1,figsize=(8,11))
    
    axes[0].hist(flatchain[:,0],bins=40,range=(5900,6300)) #temp
    axes[1].hist(flatchain[:,1],bins=40,range=(3.0,4.2)) #logg
    axes[2].hist(flatchain[:,2],bins=40,range=(40,50)) #vsini
    axes[3].hist(flatchain[:,3],bins=50,range=(25,32)) #vz
    axes[4].hist(flatchain[:,4],bins=100,range=(1e-28,2.5e-28)) #c0

    for i,ax in enumerate(axes[5:]):
        ax.hist(flatchain[:,i+5],bins=20)

    fig.subplots_adjust(hspace=0.7,top=0.95,bottom=0.06)
    #plt.show()
    plt.savefig('plots/posteriors/' + subdir + 'hist_param.png')


def joint_hist(p1,p2,**kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(flatchain[:,p1],flatchain[:,p2],**kwargs)
    plt.show()

def joint_hist_temp_log():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(flatchain[:,0],flatchain[:,1],bins=[Tbin_edges,loggbin_edges])
    ax.set_xlabel(r"$T_{\rm eff}\quad(K)$")
    ax.set_ylabel(r"$\log(g)$")
    plt.show()

def draw_chebyshev_samples():
    wl = wls[22]
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    all_inds = np.arange(len(flatchain))
    inds = np.random.choice(all_inds,size=(10,))
    ps = flatchain[inds]
    lnp = lnchain[inds]
    lnp_min, lnp_max = np.percentile(lnp, [10., 99.])
    lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
    lnp[lnp > 1.] = 1.
    lnp[lnp < 0.] = 0.
    coefss = ps[:,5:]
    for i,coefs in enumerate(coefss):
        myCh = Ch(coefs,domain=[wl[0],wl[-1]])
        c = (1.-lnp[i], 0., lnp[i])
        ax.plot(wl, myCh(wl),c=c)
    plt.show()

def plot_data(p):
    '''plots a random sample of the posterior, model, data, cheb, and residuals'''
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(11,8))

    #all_inds = np.arange(len(flatchain))
    #ind = np.random.choice(all_inds)
    #p = flatchain[ind]
    #p = np.array([5000,2.5,40,30,2.5e27])
    #lnp = lnchain[ind]
    lnp = lnprob(p)
    print(lnp)
    wlsz,flsc,fs = model_and_data(p)
    wl = wlsz[0]
    fl = flsc[0]
    f = fs[0]

    coefs = p[5:]
    myCh = Ch(coefs,domain=[wl[0],wl[-1]])
 
    ax[0].plot(wl, fl,"b")
    ax[0].plot(wl, f,"r")

    ax[1].plot(wl, myCh(wl))

    ax[2].plot(wl, fl - f)
    plt.show()

def plot_random_data():
    from model import model_and_data,lnprob
    '''plots a random sample of the posterior, model, data, cheb, and residuals'''
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(11,8))

    all_inds = np.arange(len(flatchain))
    ind = np.random.choice(all_inds)
    p = flatchain[ind]
    lnp = lnflatchain[ind]
    print(lnp)
    print(p)
    wlsz,flsc,fs = model_and_data(p)
    wl = wlsz[0]
    fl = flsc[0]
    f = fs[0]

    coefs = p[5:]
    myCh = Ch(coefs,domain=[wl[0],wl[-1]])
 
    ax[0].plot(wl, fl,"b")
    ax[0].plot(wl, f,"r")

    ax[1].plot(wl, myCh(wl))

    ax[2].plot(wl, fl - f)
    plt.show()

def staircase_plot(flatchain):
    '''flatchain has shape (N, M), where N is the number of samples and M is the number of parameters. Create a M x M staircase plot.'''
    N,M = flatchain.shape

    margin = 0.05
    bins = [np.array([5350,5450,5550,5650,5750,5850,5950,6050,6150,6250,6350,6450]),np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.0]),np.linspace(43,52,num=40),np.linspace(26,31,num=40),np.linspace(1.3e-28,2.5e-28,num=40),False,False]

    row_ax = []
    w = (1. - 2 * margin)/M
    fig = plt.figure(figsize=(8.5,8.5))
    for i in range(1,M+1):
        col_ax = []
        for j in range(1,i+1):
            L = margin + (j-1)*w
            B = 1-i*w - margin
            ax = fig.add_axes([L,B,w,w])
            if i != j:
                xbins = bins[j-1]
                ybins = bins[i-1]
                if xbins is False:
                    low,high = np.percentile(flatchain[:,j-1],[10,82])
                    xbins = np.linspace(low, high, num=40)

                if ybins is False:
                    low,high = np.percentile(flatchain[:,i-1],[10,82])
                    ybins = np.linspace(low, high, num=40)

                ax.hist2d(flatchain[:,j-1],flatchain[:,i-1],bins=[xbins,ybins])
                ax.locator_params(axis='x',nbins=8)
                ax.locator_params(axis='y',nbins=8)
            if i < M:
                ax.xaxis.set_ticklabels([])
            else:
                labels = ax.get_xticklabels() 
                for label in labels: 
                    label.set_rotation(50) 
            if j > 1:
                ax.yaxis.set_ticklabels([])

            col_ax.append(ax)
            

        row_ax.append(col_ax)

    plt.show()

def staircase_plot_thesis(flatchain):
    '''flatchain has shape (N, M), where N is the number of samples and M is the number of parameters. Create a M x M staircase plot.'''
    flatchain = flatchain[:,0:4]
    N,M = flatchain.shape

    margin = 0.1
    bins = [np.linspace(5850,6100,num=30),np.linspace(3.2,3.9,num=30),np.linspace(43,52,num=30),np.linspace(26,30,num=30)] #,np.linspace(1.3e-28,2.5e-28,num=40),False,False]

    row_ax = []
    w = (1. - 2 * margin)/M
    space = 0.05

    fig = plt.figure(figsize=(6.5,6.5))
    for i in range(1,M+1):
        col_ax = []
        for j in range(1,i+1):
            L = margin + (j-1)*w 
            B = 1-i*w - margin 
            ax = fig.add_axes([L,B,w,w])
            if i != j:
                xbins = bins[j-1]
                ybins = bins[i-1]
                ax.hist2d(flatchain[:,j-1],flatchain[:,i-1],bins=[xbins,ybins])
                ax.locator_params(axis='x',nbins=8)
                ax.locator_params(axis='y',nbins=8)
            
            if i == j:
                hbins=bins[i-1]
                if i < 4:
                    ax.hist(flatchain[:,i-1],bins=hbins)
                    ax.set_xlim(hbins[0],hbins[-1])
                if i==4:
                    ax.hist(flatchain[:,i-1],bins=hbins,orientation="horizontal")
                    ax.set_ylim(hbins[0],hbins[-1])
                
            if i < M:
                ax.xaxis.set_ticklabels([])

            else:
                labels = ax.get_xticklabels() 
                for label in labels: 
                    label.set_rotation(50) 
            if j > 1:
                ax.yaxis.set_ticklabels([])

            col_ax.append(ax)
            

        row_ax.append(col_ax)
    #bottom labels
    row_ax[-1][0].set_xlabel(r"$T_{\rm eff}$")
    row_ax[-1][0].xaxis.set_major_formatter(FSF("%.0f"))
    row_ax[-1][1].set_xlabel(r"$\log g$")
    row_ax[-1][1].xaxis.get_major_ticks()[0].label1On=False
    row_ax[-1][2].set_xlabel(r"$v \sin i$")
    row_ax[-1][2].xaxis.get_major_ticks()[-1].label1On=False
            
    #side labels
    row_ax[1][0].set_ylabel(r"$\log g$")
    row_ax[1][0].yaxis.get_major_ticks()[-1].label1On=False
    row_ax[1][0].yaxis.get_major_ticks()[0].label1On=False
    row_ax[2][0].set_ylabel(r"$v \sin i$")
    row_ax[3][0].set_ylabel(r"$v_z$")
    

    #plt.show()
    fig.savefig("plots/staircase.eps")

def test_hist():
    bins = np.array([5350,5450,5550,5650,5750,5850,5950,6050,6150,6250,6350,6450])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(flatchain[:,0],bins=bins)
    ax.set_xlim(bins[0],bins[-1])
    plt.show()

def plot_walker_position():
    nwalkers = chain.shape[0]
    nsteps = chain.shape[1]
    steps = np.arange(nsteps)
    for param in range(nparams):
        fig = plt.figure(figsize=(11,11))
        ax = fig.add_subplot(111)
        for walker in range(nwalkers):
            ax.plot(steps,chain[walker,:,param])
        fig.savefig("plots/walkers/{:0>2.0f}.png".format(param))

def get_acor():
    for param in range(nparams):
        print(acor.acor(chain[:,:,param]))


#print(len(flatchain))
#hist_param(flatchain[590000:])
hist_param(flatchain)
#plot_random_data()
#joint_hist(2,3,bins=[20,40],range=((50,65),(28,31)))
#joint_hist(0,4,range=((),()))
#draw_chebyshev_samples()
#staircase_plot_thesis(flatchain[590000:])
#test_hist()
#plot_walker_position()
#get_acor()
