#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Chebyshev as Ch
from matplotlib.ticker import FormatStrFormatter as FSF
#import acor
import model as m
import yaml
import os

confname = 'config.yaml' #sys.argv[1]
f = open(confname)
config = yaml.load(f)
f.close()

wr = config['walker_ranges']
nparams = config['nparams']
norders = len(config['orders'])


def auto_hist_param_linspace(flatchain):
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 11))
    labels = [r"$T_{\rm eff}$ (K)", r"$\log g$ (dex)", r'$Z$ (dex)', r"$v \sin i$ (km/s)", r"$v_z$ (km/s)", r"$A_v$ (mag)", r"$R^2/d^2$" ]

    axes[0].hist(flatchain[:, 0], bins=np.linspace(wr['temp'][0],wr['temp'][1],num=40)) #temp
    axes[1].hist(flatchain[:, 1], bins=np.linspace(wr['logg'][0],wr['logg'][1],num=40)) #logg
    axes[2].hist(flatchain[:, 2], bins=np.linspace(wr['Z'][0],wr['Z'],num=40)) #Z
    axes[3].hist(flatchain[:, 3], bins=np.linspace(wr['vsini'][0], wr['vsin'][1],num=40)) #vsini
    axes[4].hist(flatchain[:, 4], bins=np.linspace(wr['vz'][0], wr['vz'][1],num=40)) #vz
    axes[5].hist(flatchain[:,5], bins=np.linspace(wr['Av'][0], wr['Av'][1],num=40)) # Av
    axes[6].hist(flatchain[:,6], bins=np.linspace(wr['flux_factor'][0],wr['flux_factor'][1], num=40))#, range=(1e-28,1e-27)) #fluxfactor

    fig.subplots_adjust(hspace=0.9, top=0.95, bottom=0.06)
    plt.savefig(config['output_dir'] + '/' + config['name'] + '/hist_param.png')

def auto_hist_param(flatchain):
    '''Just a simple histogram with no care about bin number, sizes, or location.'''
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 11))
    labels = [r"$T_{\rm eff}$ (K)", r"$\log g$ (dex)", r'$Z$ (dex)', r"$v \sin i$ (km/s)", r"$v_z$ (km/s)", r"$A_v$ (mag)", r"$R^2/d^2$" ]

    axes[0].hist(flatchain[:, 0]) #temp
    axes[1].hist(flatchain[:, 1]) #logg
    axes[2].hist(flatchain[:, 2]) #Z
    axes[3].hist(flatchain[:, 3]) #vsini
    axes[4].hist(flatchain[:, 4]) #vz
    axes[5].hist(flatchain[:,5]) # Av
    axes[6].hist(flatchain[:,6]) #fluxfactor

    for i in range(7):
        axes[i].set_xlabel(labels[i])

    fig.subplots_adjust(hspace=0.9, top=0.95, bottom=0.06)
    plt.savefig(config['output_dir'] + '/' + config['name'] + '/hist_param.png')

def hist_nuisance_param(flatchain):
    #make a nuisance directory
    #Create necessary output directories using os.mkdir, if it does not exist
    nuisance_dir = config['output_dir'] + "/" + config['name'] + '/nuisance/'
    if not os.path.exists(nuisance_dir):
        os.mkdir(nuisance_dir)
        print("Created output directory", nuisance_dir)
    else:
        print(nuisance_dir, "already exists, overwriting.")

    cs = flatchain[:,nparams:]
    print(cs.shape)

    if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == "lnprob_gaussian"):
        for i in range(norders):
            HEAD = i * 4
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 11))
            axes[0].hist(cs[:,HEAD + 0])
            axes[1].hist(cs[:,HEAD + 1])
            axes[2].hist(cs[:,HEAD + 2])
            axes[3].hist(cs[:,HEAD + 3])
            fig.savefig(nuisance_dir + "{order:0>2.0f}.png".format(order=config['orders'][i]+1))

    if (config['lnprob'] == 'lnprob_gaussian_marg') or (config['lnprob'] == 'lnprob_lognormal_marg'):
        pass


def joint_hist(p1, p2, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(flatchain[:, p1], flatchain[:, p2], **kwargs)
    plt.show()


def joint_hist_temp_log():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(flatchain[:, 0], flatchain[:, 1], bins=[Tbin_edges, loggbin_edges])
    ax.set_xlabel(r"$T_{\rm eff}\quad(K)$")
    ax.set_ylabel(r"$\log(g)$")
    plt.show()


def draw_chebyshev_samples():
    wl = wls[22]
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    all_inds = np.arange(len(flatchain))
    inds = np.random.choice(all_inds, size=(10,))
    ps = flatchain[inds]
    lnp = lnchain[inds]
    lnp_min, lnp_max = np.percentile(lnp, [10., 99.])
    lnp = (lnp - lnp_min) / (lnp_max - lnp_min)
    lnp[lnp > 1.] = 1.
    lnp[lnp < 0.] = 0.
    coefss = ps[:, 5:]
    for i, coefs in enumerate(coefss):
        myCh = Ch(coefs, domain=[wl[0], wl[-1]])
        c = (1. - lnp[i], 0., lnp[i])
        ax.plot(wl, myCh(wl), c=c)
    plt.show()

def visualize_draws(flatchain, lnflatchain, sample_num=10):
    visualize_dir = config['output_dir'] + "/" + config['name'] + '/visualize/'
    if not os.path.exists(visualize_dir):
        os.mkdir(visualize_dir)
        print("Created output directory", visualize_dir)
    else:
        print(visualize_dir, "already exists, overwriting.")

    all_inds = np.arange(len(flatchain))

    for i in range(sample_num):
        #For each sample from the posterior

        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        #Choose a random sample from the chain
        ind = np.random.choice(all_inds)
        p = flatchain[ind]
        lnp = lnflatchain[ind]
        sample_dir = visualize_dir + 'sample{i:0>2.0f}/'.format(i=i)
        #write p and lnp to a file in sample dir
        f = open('params.txt',"w")
        f.write("Parameters: %s \n" % (p,))
        f.write("lnp: %s \n" % (lnp,))
        f.close()

        #Get wavelength, flux, and sigmas
        wl = m.wls[order]
        fl = m.fls[order]
        sigma = m.sigmas[order]

        #Reproduce the model spectrum for that parameter combo
        wlsz = m.shift_TRES(2.)
        f = m.model(wlsz, 5900., 3.5, 0.0, 5., 0.0, 1e-10)[order]

        for order in config['orders']:
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(11, 8),sharex=True)



            #coefs = p[m.config['nparams']:]
            #coefs_arr = coefs.reshape(m.config.len(orders), -1)
            #print(coefs_arr)

            ax[0].plot(wl, fl, "b")
            ax[0].plot(wl, f, "r")
            ax[0].fill_between(wl, fl - sigma, fl + sigma, color="0.5", alpha=0.8)


            ax[2].plot(wl, fl - f)
            ax[2].fill_between(wl, - sigma, sigma, color="0.5", alpha=0.8)
            ax[2].set_xlim(wl[0],wl[-1])
            plt.show()

            plt.hist(sigma)
            plt.show()
            plt.hist(fl-f)
            plt.show()




def plot_data_and_residuals():
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(11, 8),sharex=True)

    wl = m.wls[0]
    fl = m.fls[0]
    sigma = m.sigmas[0]

    wlsz = m.shift_TRES(2.)
    f = m.model(wlsz, 5900., 3.5, 0.0, 5., 0.0, 1e-10)[0]

    #coefs = p[m.config['nparams']:]
    #coefs_arr = coefs.reshape(m.config.len(orders), -1)
    #print(coefs_arr)

    ax[0].plot(wl, fl, "b")
    ax[0].plot(wl, f, "r")
    ax[0].fill_between(wl, fl - sigma, fl + sigma, color="0.5", alpha=0.8)


    ax[2].plot(wl, fl - f)
    ax[2].fill_between(wl, - sigma, sigma, color="0.5", alpha=0.8)
    ax[2].set_xlim(wl[0],wl[-1])
    plt.show()

    plt.hist(sigma)
    plt.show()
    plt.hist(fl-f)
    plt.show()


def plot_random_data():
    from model import model_and_data,lnprob_old
    import model as m

    '''plots a random sample of the posterior, model, data, cheb, and residuals'''
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 6),sharex=True)

    all_inds = np.arange(len(flatchain))
    ind = np.random.choice(all_inds)

    p = flatchain[ind]
    lnp = lnflatchain[ind]

    wlsz, flsc, fs = model_and_data(p)
    wl = wlsz[0]
    fl = flsc[0]
    f = fs[0]
    sigma = m.sigmas[0]

    lnp_calc = lnprob_old(p)

    print("lnprob", lnp)
    print("chi^2_red", -2. * lnp/len(wl))
    print("lnpcalc", lnp_calc)
    print("chi^2_calc", -2 * lnp_calc/len(wl))
    print("Parameters", p)

    coefs = p[5:]
    myCh = Ch(np.append([1], coefs), domain=[wl[0], wl[-1]])

    ax[0].plot(wl, fl, "b")
    ax[0].plot(wl, f, "r")
    ax[0].fill_between(wl, fl - sigma, fl + sigma, color="0.5", alpha=0.8)

    ax[1].plot(wl, myCh(wl))

    ax[2].plot(wl, fl - f)
    ax[2].fill_between(wl, - sigma, sigma, color="0.5", alpha=0.8)
    ax[2].set_xlim(wl[0],wl[-1])
    plt.show()


def staircase_plot(flatchain):
    '''flatchain has shape (N, M), where N is the number of samples and M is the number of parameters. Create a M x M
    staircase plot.'''
    N, M = flatchain.shape

    margin = 0.05
    bins = [np.array([5350, 5450, 5550, 5650, 5750, 5850, 5950, 6050, 6150, 6250, 6350, 6450]),
            np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.0]),
            np.linspace(43, 52, num=40), np.linspace(26, 31, num=40), np.linspace(1.3e-28, 2.5e-28, num=40), False,
            False]

    row_ax = []
    w = (1. - 2 * margin) / M
    fig = plt.figure(figsize=(8.5, 8.5))
    for i in range(1, M + 1):
        col_ax = []
        for j in range(1, i + 1):
            L = margin + (j - 1) * w
            B = 1 - i * w - margin
            ax = fig.add_axes([L, B, w, w])
            if i != j:
                xbins = bins[j - 1]
                ybins = bins[i - 1]
                if xbins is False:
                    low, high = np.percentile(flatchain[:, j - 1], [10, 82])
                    xbins = np.linspace(low, high, num=40)

                if ybins is False:
                    low, high = np.percentile(flatchain[:, i - 1], [10, 82])
                    ybins = np.linspace(low, high, num=40)

                ax.hist2d(flatchain[:, j - 1], flatchain[:, i - 1], bins=[xbins, ybins])
                ax.locator_params(axis='x', nbins=8)
                ax.locator_params(axis='y', nbins=8)
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
    '''flatchain has shape (N, M), where N is the number of samples and M is the number of parameters. Create a M x M
    staircase plot.'''
    flatchain = flatchain[:, 0:4]
    N, M = flatchain.shape

    margin = 0.1
    bins = [np.linspace(5850, 6100, num=30), np.linspace(3.2, 3.9, num=30), np.linspace(43, 52, num=30),
            np.linspace(26, 30, num=30)] #,np.linspace(1.3e-28,2.5e-28,num=40),False,False]

    row_ax = []
    w = (1. - 2 * margin) / M
    space = 0.05

    fig = plt.figure(figsize=(6.5, 6.5))
    for i in range(1, M + 1):
        col_ax = []
        for j in range(1, i + 1):
            L = margin + (j - 1) * w
            B = 1 - i * w - margin
            ax = fig.add_axes([L, B, w, w])
            if i != j:
                xbins = bins[j - 1]
                ybins = bins[i - 1]
                ax.hist2d(flatchain[:, j - 1], flatchain[:, i - 1], bins=[xbins, ybins])
                ax.locator_params(axis='x', nbins=8)
                ax.locator_params(axis='y', nbins=8)

            if i == j:
                hbins = bins[i - 1]
                if i < 4:
                    ax.hist(flatchain[:, i - 1], bins=hbins)
                    ax.set_xlim(hbins[0], hbins[-1])
                if i == 4:
                    ax.hist(flatchain[:, i - 1], bins=hbins, orientation="horizontal")
                    ax.set_ylim(hbins[0], hbins[-1])

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
    row_ax[-1][1].xaxis.get_major_ticks()[0].label1On = False
    row_ax[-1][2].set_xlabel(r"$v \sin i$")
    row_ax[-1][2].xaxis.get_major_ticks()[-1].label1On = False

    #side labels
    row_ax[1][0].set_ylabel(r"$\log g$")
    row_ax[1][0].yaxis.get_major_ticks()[-1].label1On = False
    row_ax[1][0].yaxis.get_major_ticks()[0].label1On = False
    row_ax[2][0].set_ylabel(r"$v \sin i$")
    row_ax[3][0].set_ylabel(r"$v_z$")


    #plt.show()
    fig.savefig("plots/staircase.eps")

def staircase_plot_proposal(flatchain):
    '''flatchain has shape (N, M), where N is the number of samples and M is the number of parameters. Plot only the T_eff vs. log g'''


    #Do a 2D histogram, get values
    #Set a standard deviation, encloses 63%, 97, 99 or whatever.

    plt.hexbin(flatchain[:,0],flatchain[:,1])
    plt.show()
    margin = 0.1
    bins = [np.linspace(3000, 4000, num=30), np.linspace(3.5, 4.5, num=30)] #left most edges of bins except [-1], which is right edge
    margin = 0.05

    fig = plt.figure(figsize=(6.5, 6.5))

    H = np.histogramdd(flatchain[:,0:2])#,bins=bins)
    H_mod = H[0] #2D array with hist on bins
    Hshape = H_mod.shape

    #flatten array
    H_flat = H_mod.flatten()

    #do argsort, cumsum, find values
    args = np.argsort(H_flat)[::-1]
    iargs = np.argsort(args)

    #print(args)

    H_sort = H_flat[args] #this array is ranked from highest pixel to lowest pixel
    stdvs = np.cumsum(H_sort)/np.sum(H_sort) #the array so we go outwards from the most dense point to the least, and normalize

    #print(stdvs)

    sig_bounds = np.array([0.6827, 0.9545 , 0.9973])

    #find all values where stdvs < sig_bounds
    sig_1 = stdvs <= sig_bounds[0] #darkest color
    sig_2 = (stdvs > sig_bounds[0]) & (stdvs <= sig_bounds[1])
    sig_3 = (stdvs > sig_bounds[1]) & (stdvs <= sig_bounds[2])
    sig_4 = (stdvs > sig_bounds[2]) #lightest color or white
    #print(sig_1,sig_2,sig_3,sig_4)

    colors = np.array([(215, 48, 31),(252, 141, 89),(253, 204, 138),(254, 240, 217)])/256. #goes from darkest to lightest
    #colors = np.array([(43,140,190),(166,189,219),(236,231,242),(256,256,256)])/256.
    #colors = np.array([(227, 74, 51),(253, 187, 132),(254, 232, 200),(256, 256, 256)])/256. #goes from darkest to lightest

    H_plot = np.empty((len(H_sort),3))

    H_plot[sig_1[iargs]] = colors[0]
    H_plot[sig_2[iargs]] = colors[1]
    H_plot[sig_3[iargs]] = colors[2]
    H_plot[sig_4[iargs]] = colors[3]

    H_plot = H_plot.reshape(np.append(Hshape,3))

    ax = fig.add_subplot(111)
    ax.imshow(H_plot,origin='lower',extent=[bins[0][0],bins[0][-1],bins[1][0],bins[1][-1]],aspect='auto',interpolation='none')
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_xlabel(r"$T_{\rm eff}$")
    ax.set_ylabel(r"$\log g$")


    #plt.show()
    fig.savefig("plots/staircase_mini.eps")

def mini_hist(data,label1=r"$x_1$", label2=r"$x_2$",bins=None):
    '''Data comes as a N,D array, where N is the number of samples and D=2. The first column goes on the x-axis and the second column goes on the y-axis.'''

    if bins==None:
        #Auto-center
        #Determine bins based upon mean and std-dev preliminarily
        Hp,binsp = np.histogramdd(data)
        Hp = Hp/np.max(Hp)
        max = np.unravel_index(np.argmax(Hp),dims=Hp.shape)

        mu_x1 = np.average(binsp[0][max[0]:max[0]+1])
        mu_x2 = np.average(binsp[1][max[1]:max[1]+1])
        print(mu_x1,mu_x2)
        std_x1 = np.std(data[:,0])
        std_x2 = np.std(data[:,1])
        print(std_x1, std_x2)

        #Re-calculated bins based upon mu and std_dev
        bins = [np.linspace(mu_x1 - 4 * std_x1, mu_x1 + 4 * std_x1,num=20),
                np.linspace(mu_x2 - 4 * std_x2, mu_x2 + 4 * std_x2,num=20)]

    H,bins = np.histogramdd(data,bins=bins)
    #Scale H to max
    H = H/np.max(H)
    Hshape = H.shape

    #flatten array
    H_flat = H.flatten()

    sig_heights = np.array([0.0111,0.135, 0.607])

    #find all values where stdvs < sig_heights
    sig_4 = H_flat <= sig_heights[0] #darkest color
    sig_3 = (H_flat > sig_heights[0]) & (H_flat <= sig_heights[1])
    sig_2 = (H_flat > sig_heights[1]) & (H_flat <= sig_heights[2])
    sig_1 = (H_flat > sig_heights[2]) #lightest color or white
    #print(sig_1,sig_2,sig_3,sig_4)

    colors = np.array([(215, 48, 31), (252, 141, 89), (253, 204, 138), (254, 240, 217)]) / 256. #from dark to light

    H_plot = np.empty((len(H_flat), 3))

    H_plot[sig_1] = colors[0]
    H_plot[sig_2] = colors[1]
    H_plot[sig_3] = colors[2]
    H_plot[sig_4] = colors[3]

    H_plot = H_plot.reshape(np.append(Hshape, 3))
    H_plot = np.transpose(H_plot, axes=(1,0,2))

    fig = plt.figure(figsize=(3, 3))

    L = 0.20
    w_center = 0.6
    B = 0.20
    w_side = 0.15
    w_sep = 0.01

    ax_center = fig.add_axes([L, B, w_center, w_center])
    ax_center.imshow(H_plot, origin='lower', aspect='auto',interpolation='none', extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]])

    ax_top = fig.add_axes([L, B + w_center + w_sep, w_center, w_side])
    ax_top.hist(data[:,0],bins=bins[0],color='w')
    ax_top.set_xlim(bins[0][0],bins[0][-1])
    ax_top.xaxis.set_ticklabels([])
    ax_top.yaxis.set_ticklabels([])

    ax_side = fig.add_axes([L + w_center + w_sep, B, w_side, w_center])
    ax_side.hist(data[:, 1], bins=bins[1],orientation='horizontal',color='w')
    ax_side.set_ylim(bins[1][0], bins[1][-1])
    ax_side.xaxis.set_ticklabels([])
    ax_side.yaxis.set_ticklabels([])
    #ax.contour(H,levels=sig_heights, extent=[bins[1][0], bins[1][-1], bins[0][0], bins[0][-1]])
    #ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax_center.set_xlabel(label1)
    ax_center.set_ylabel(label2)

    ax_center.xaxis.set_major_formatter(FSF("%.0f"))
    xlabels = ax_center.get_xticklabels()
    for label in xlabels:
        label.set_rotation(50)
    fig.savefig('plots/staircase_mini.eps')



def plot_walker_position():
    nwalkers = chain.shape[0]
    nsteps = chain.shape[1]
    steps = np.arange(nsteps)
    for param in range(nparams):
        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111)
        for walker in range(nwalkers):
            ax.plot(steps, chain[walker, :, param])
        fig.savefig("plots/walkers/{:0>2.0f}.png".format(param))


#def get_acor():
#    for param in range(nparams):
#        print(acor.acor(chain[:, :, param]))


def main():
    #What to do if you're just running this based off of config
    chain = np.load("output/" + config['name'] + "/chain.npy")
    nwalkers = chain.shape[0]
    nsteps = chain.shape[1]

    #Give flatchain, too
    s = chain.shape
    flatchain = chain.reshape(s[0] * s[1], s[2])
    #flatchain = flatchain[80000:]
    lnchain = np.load("output/" + config['name'] + "/lnprobchain.npy")
    lnflatchain = lnchain.flatten()
    ndim_chain = flatchain.shape[1]

    auto_hist_param(flatchain)
    hist_nuisance_param(flatchain)
    visualize_draws(flatchain)
    #plot_random_data()
    #plot_random_data()
    #plot_data_and_residuals()
    #joint_hist(2,3,bins=[20,40],range=((50,65),(28,31)))
    #joint_hist(0,4,range=((),()))
    #draw_chebyshev_samples()
    #staircase_plot_thesis(flatchain[590000:])
    #staircase_plot_proposal(flatchain)
    #test_hist()
    #plot_walker_position()
    #get_acor()

    #x1 = np.random.normal(0,1,size=(10000,))
    #x2 = np.random.normal(0,3,size=(10000,))
    #data = np.array([x1,x2]).T #same format as flatchain
    #mini_hist(flatchain[:,0:2],r"$T_{\rm eff}$",r"$\log g$",bins=[np.linspace(4750,4950,num=30),np.linspace(4.0,4.4,num=30)])

if __name__=="__main__":
    main()