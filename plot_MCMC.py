import asciitable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import MCMC

#data1 = asciitable.read("run1.dat")
#data2 = asciitable.read("run2.dat")
#data3 = asciitable.read("run3.dat")
#dataa = asciitable.read("runa.dat")
#datab = asciitable.read("runb.dat")
data = asciitable.read("run_0.dat")
#data_tuple = (data1,data2,data3,dataa,datab,datac)

def plot_vs_j(data):
    fig, (axA,axB,axC,axW) = plt.subplots(4,1, sharex=True,figsize=(4,6))
    axA.plot(data['j'],data['A'])
    axA.set_ylabel("A")
    axA.locator_params(axis='y',nbins=6)
    axB.plot(data['j'],data['B'])
    axB.set_ylabel("B")
    axB.locator_params(axis='y',nbins=6)
    axC.plot(data['j'],data['C'])
    axC.set_ylabel('C')
    axC.locator_params(axis='y',nbins=6)
    axW.plot(data['j'],data['omega'])
    axW.locator_params(axis='y',nbins=4)
    #y_axis = axW.get_yaxis()
    #SF = matplotlib.ticker.ScalarFormatter()#useOffset=True, useMathText=None, useLocale=None)
    #SF.set_scientific(True)
    #y_axis.set_major_formatter(SF)
    axW.set_ylabel(r'$\omega$')
    axW.set_xlabel(r'$j$')

    plt.show()
    #plt.savefig('param_vs_j.eps')

def plot_2d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data['m'],data['b'],'b',lw=0.2)
    ax.plot(data['m'],data['b'],"bo")
    ax.set_xlabel("m")
    ax.set_ylabel("b")
    #ax.fill([0,0,0.55,0.55],[0,0.5,0.5,0],hatch="/",fill=False)
    #ax.annotate("Start",(data['a'][0]-0.09,data['b'][0]-0.01))
    #ax.annotate("Burn in region",(0.1,0.1))
    #fig.savefig("2d_burn_in.eps")
    plt.show()

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

def prob_bad_points(data):
    burn_in = 10000
    ms = data["m"][burn_in:]
    bs = data["b"][burn_in:]
    Pbs = data["Pb"][burn_in:]
    Ybs = data["Yb"][burn_in:]
    Vbs = data["Vb"][burn_in:]
    N = len(ms)
    indexes = np.arange(0,len(xi),1)
    probs = []
    for i in indexes:
        foreground = 1./N * np.sum((1-Pbs)/(np.sqrt(2. * np.pi) * sigma[i]) * np.exp(-(yi[i] - ms * xi[i] - bs)**2/(2. * sigma[i]**2)))
        background = 1./N * np.sum(Pbs/np.sqrt(2. * np.pi * (Vbs + sigma[i]**2)) * np.exp(-(yi[i] - Ybs)**2/(2. * (Vbs + sigma[i]**2))))
        prob = background/foreground
        probs.append(prob)
        #print(i,"%.2e" % prob)
    plt.semilogy(indexes,probs,"o")
    plt.ylabel(r"$p(q_i=0)$")
    plt.xlabel(r"$i$")
    plt.show()


def hist_param(data):
    burn_in = 10000
    fig, (axm,axb,axP,axY,axV) = plt.subplots(5,1,figsize=(4,7))
    axm.hist(data["m"][burn_in:],normed=True)
    axm.set_xlabel("m")
    axm.set_ylabel("Percentage")

    axb.hist(data["b"][burn_in:],normed=True)
    axb.set_xlabel("b")
    axb.set_ylabel("Percentage")

    axP.hist(data["Pb"][burn_in:],normed=True)
    axP.set_xlabel("P")
    axP.set_ylabel("Percentage")
    axP.locator_params(axis='y',nbins=8)

    axY.hist(data["Yb"][burn_in:],normed=True)
    axY.set_xlabel("Y")
    axY.set_ylabel("Percentage")
    axY.locator_params(axis='y',nbins=8)

    axV.hist(data["Vb"][burn_in:],normed=True)
    axV.set_xlabel("V")
    axV.set_ylabel("Percentage")
    axV.locator_params(axis='y',nbins=8)

    fig.subplots_adjust(hspace=0.5,top=0.95,bottom=0.06)
    plt.show()
    #plt.savefig('hist_param.svg')


def joint_hist(data):
    burn_in = 10000
    m = data["m"][burn_in:]
    b = data["b"][burn_in:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hexbin(m,b,gridsize=50)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$b$")
    plt.show()

plot_vs_j(datac)
#calc_mean_and_sigma(datab)
#hist_param(datab)
#prob_bad_points(datab)
#hist_param(data1)
#plot_2d(datab)
#joint_hist(dataa)
#plot_2d_all()
#for data in data_tuple:
#    mean_a,mean_b = calc_mean_and_sigma(data)
#    print(MCMC.chi_sq(mean_a,mean_b,0.1))
#    print()
