import numpy as np
import emcee
import sys
import yaml
import plot_MCMC
from matplotlib.ticker import FormatStrFormatter as FSF
import model as m
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import fmin

if len(sys.argv) > 1:
    confname= sys.argv[1]
else:
    confname = 'config.yaml'
f = open(confname)
config = yaml.load(f)
f.close()

nwalkers = config['pnwalkers']
ncoeff = config['ncoeff']
norders = len(config['orders'])
wr = config['walker_ranges']

if (config['lnprob'] == 'lnprob_gaussian_marg') or (config['lnprob'] == 'lnprob_lognormal_marg'):
    ndim = config['nparams'] + norders
if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == "lnprob_gaussian") \
    or (config['lnprob'] == 'lnprob_mixed'):
    ndim = config['nparams'] + ncoeff * norders

lnprob = getattr(m, config['lnprob'])


fig, ax = plt.subplots(nrows=2, figsize=(8,10), sharex=True)
plt.subplots_adjust(left=0.05, bottom=0.55, top=0.96, right=0.95)

wlsz, f, k, flatchain = m.model_p(np.array([6000, 4.0, 0.0, 5.0, 15, 0.0, 2e-20, 1, -0.02, -0.019, -0.002 ]))
l0, = ax[0].plot(wlsz[0], m.fls[0], lw=2, color='blue')
l, = ax[0].plot(wlsz[0], f[0], lw=2, color='red')
residuals = (m.fls[0] - f[0])/m.sigmas[0]
r, = ax[1].plot(wlsz[0], residuals)
ax[1].xaxis.set_major_formatter(FSF("%.2f"))

#create slider bars
left = 0.15
width = 0.65
height = 0.03

stemp = Slider(plt.axes([left, 0.5, width, height]) , 'Temp', 5000, 7000, valinit=6000)
slogg = Slider(plt.axes([left, 0.45, width, height]), r'$\log g$', 3.0, 5.0, valinit=4.0)
sZ = Slider(plt.axes([left, 0.40, width, height]), r'$Z$', -1.0, 0.5, valinit=0.0)
svsini = Slider(plt.axes([left, 0.35, width, height]), r'$v \sin i$', 1.0, 10, valinit=5.0)
svz = Slider(plt.axes([left, 0.30, width, height]), r'$v_z$', 14, 16, valinit=15.0)
sAv = Slider(plt.axes([left, 0.25, width, height]), r'$A_v$', 0.0, 1, valinit=0.0)
sff = Slider(plt.axes([left, 0.20, width, height]), r'$ff$', 1.8e-20, 2.2e-20, valinit=2.0e-20)
sc1 = Slider(plt.axes([left, 0.15, width, height]), r'$c_1$', -0.1, 0.1, valinit=-0.02)
sc2 = Slider(plt.axes([left, 0.1, width, height]), r'$c_2$', -0.1, 0.1, valinit=-0.019)
sc3 = Slider(plt.axes([left, 0.05, width, height]), r'$c_3$', -0.1, 0.1, valinit=-0.002)

def mixed(p):
    func = lambda x: p[0] * (np.exp(-0.5 * (x - p[2])**2/p[3]**2) + p[1] * np.exp(-np.abs(x - p[4])/p[5]))
    return func

fig2 = plt.figure(figsize=(5,5))
ax2 = fig2.add_subplot(111)

n, bins = np.histogram(residuals, bins=40)
n = n/np.max(n)
bin_centers = (bins[:-1] + bins[1:])/2
var = n.copy()
var[var == 0] = 1.
lhist, = ax2.plot(bin_centers, n, "o")
ax2.set_title("Histogram of Residuals")
ax2.set_xlabel(r"$\sigma$ (Poisson)")

xs = np.linspace(-20, 20, num = 150)
mix = mixed([1, 0.2, 0, 3, 0, 3])
lmix, = ax2.plot(xs, mix(xs))




def update(val):
    T = stemp.val
    G = slogg.val
    Z = sZ.val
    vsini = svsini.val
    vz = svz.val
    Av = sAv.val
    ff = sff.val
    c1 = sc1.val
    c2 = sc2.val
    c3 = sc3.val

    #run a partial chain on these parameters
    #lnprob_partial = m.wrap_lnprob(lnprob, T, G, Z, vsini)
    #func = lambda p: - lnprob_partial(p)
    #new_params = fmin(func, np.array([15, 0.0, 2e-20, 1, 0, 0, 0]))
    #print(new_params)
    #lnprobfn = emcee.ensemble._function_wrapper(lnprob_partial, None)
    #new_params = sample(lnprob_partial)
    #new_p = np.hstack((np.array([T, G, Z, vsini]), new_params))
    #print(new_p)

    wlsz, f, k, flatchain = m.model_p(np.array([T, G, Z, vsini, vz, Av, ff, 1.0, c1, c2, c3]))
    lnprob_val = m.lnprob_mixed(np.array([T, G, Z, vsini, vz, Av, ff, 1.0, c1, c2, c3]))
    print(lnprob_val)
    ax[0].annotate("lnprob: {:.1f}".format(lnprob_val), (0.8, 0.1), xycoords='axes fraction', backgroundcolor='w')
    l0.set_xdata(wlsz[0])
    l.set_xdata(wlsz[0])
    l.set_ydata(f[0])

    r.set_xdata(wlsz[0])
    residuals = (m.fls[0] - f[0])/m.sigmas[0]
    r.set_ydata(residuals)

    n, bins = np.histogram(residuals, bins=40)
    n = n/np.max(n)
    bin_centers = (bins[:-1] + bins[1:])/2
    var = n.copy()
    var[var == 0] = 1.

    mixed_func = lambda p: np.sum((n - mixed(p)(bin_centers))**2/var)

    mparam = fmin(mixed_func, [1, 0.2, 0, 3, 0, 3])
    print("Mixture parameters", mparam)
    mix = mixed(mparam)
    lmix.set_ydata(mix(xs))

    ax2.annotate(r"$\mu_G$:{:.1f}    $\sigma_G$:{:.1f}".format(mparam[2], mparam[3]),
                 (0.1, 0.9), xycoords='axes fraction', backgroundcolor='w')
    ax2.annotate(r"$\mu_E$:{:.1f}    $\sigma_E$:{:.1f}".format(mparam[4], mparam[5]),
                 (0.1, 0.8), xycoords='axes fraction', backgroundcolor='w')
    ax2.annotate(r"$A_E$:{:.1f}".format(mparam[1]),
                 (0.1, 0.7), xycoords='axes fraction', backgroundcolor='w')

    lhist.set_xdata(bin_centers)
    lhist.set_ydata(n)

    fig.canvas.draw_idle()
    fig2.canvas.draw_idle()

stemp.on_changed(update)
slogg.on_changed(update)
sZ.on_changed(update)
svsini.on_changed(update)
svz.on_changed(update)
sAv.on_changed(update)
sff.on_changed(update)
sc1.on_changed(update)
sc2.on_changed(update)
sc3.on_changed(update)


outdir = 'output/partial/'

def generate_nuisance_params():
    '''convenience method for generating walker starting positions for nuisance parameters.
    Reads number of orders from config, type of lnprob and generates c0, c1, c2, c3 locations,
    or just c0 locations if lnprob_marg'''
    norders = len(config['orders'])
    c0s = np.random.uniform(low=wr['c0'][0], high = wr['c0'][1], size=(norders, nwalkers))
    if (config['lnprob'] == 'lnprob_gaussian_marg') or (config['lnprob'] == 'lnprob_lognormal_marg'):
        return c0s

    if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == "lnprob_gaussian") \
        or (config['lnprob'] == 'lnprob_mixed'):
        #do this for each order. create giant array for cns, then do a stride on every c0 to replace them.
        cs = np.random.uniform(low=wr['cs'][0], high = wr['cs'][1], size=(ncoeff*norders, nwalkers))
        cs[::ncoeff] = c0s
        return cs



def sample(lnprob_partial):

    sampler = emcee.EnsembleSampler(nwalkers, ndim - 4, lnprob_partial) #, threads=config['threads'])

    # Choose an initial set of walker parameter positions, randomly distributed across a reasonable range of parameters.
    vz = np.random.uniform(low=wr['vz'][0], high=wr['vz'][1], size=(nwalkers,))
    Av = np.random.uniform(low=wr['Av'][0], high = wr['Av'][1], size=(nwalkers,))
    flux_factor = np.random.uniform(low=wr['flux_factor'][0], high=wr['flux_factor'][1], size=(nwalkers,))
    cs = generate_nuisance_params()

    p0 = np.vstack((np.array([vz, Av, flux_factor]), cs)).T #Stack cs onto the end

    pos, prob, state = sampler.run_mcmc(p0, config['pburn_in'])

    print("Burned in chain")
    sampler.reset()

    sampler.run_mcmc(pos, config['piterations'], rstate0=state)

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker.
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    #write chains to file
    #np.save(outdir + "chain.npy", sampler.chain)
    #np.save(outdir + "lnprobchain.npy", sampler.lnprobability)
    #np.save(outdir + "flatchain.npy", sampler.flatchain)
    #np.save(outdir + "flatlnprobchain.npy", sampler.flatlnprobability)

    new_params = np.mean(sampler.flatchain, axis=0)
    #print(new_params)
    return new_params

    #if config['plots'] == True:
    #    plot_MCMC.auto_hist_param(sampler.flatchain)
    #    plot_MCMC.hist_nuisance_param(sampler.flatchain)

# To improve
# do MCMC behind scenes to determine best fit c0 coefficients
# Plot best fit coefficients on the screen as well
# Plot lnprob with each spectrum
# highlight 1 sigma residuals


plt.show()