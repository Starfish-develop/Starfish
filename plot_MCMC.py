#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

flatchain = np.load("flatchain.npy")
nparams = flatchain.shape[1]

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
Ts = T_points[25:47]
Tbin_edges = np.diff(Ts)/2 + Ts[:-1]
loggbin_edges = [0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.0]

def hist_param(flatchain):
    fig, axes = plt.subplots(nrows=nparams,ncols=1,figsize=(8,11))
    
    axes[0].hist(flatchain[:,0],Tbin_edges) #temp
    axes[1].hist(flatchain[:,1],loggbin_edges) #logg
    axes[2].hist(flatchain[:,2],bins=20,range=(35,70)) #vsini
    axes[3].hist(flatchain[:,3],bins=50,range=(25,32)) #vz
    axes[4].hist(flatchain[:,4],bins=100,range=(8e26,3e27)) #c0

    for i,ax in enumerate(axes[5:]):
        ax.hist(flatchain[:,i+5],bins=20,range=(-5,5))

    fig.subplots_adjust(hspace=0.7,top=0.95,bottom=0.06)
    #plt.show()
    plt.savefig('hist_param.png')


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

#plot_vs_j(data)
#calc_mean_and_sigma(datab)
#hist_param(datab)
#prob_bad_points(datab)
hist_param(flatchain)
#joint_hist(2,3,bins=[20,40],range=((50,65),(28,31)))
#joint_hist(0,4,range=((),()))
#plot_2d(datab)
#joint_hist(dataa)
#plot_2d_all()
