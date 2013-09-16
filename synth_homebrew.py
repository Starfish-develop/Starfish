import numpy as np
import matplotlib.pyplot as plt
from PHOENIX_tools import load_flux_full,w_full
from deredden import deredden
from astropy.io import fits
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from PHOENIX_tools import load_flux_full,w_full
from matplotlib.ticker import FormatStrFormatter as FSF

base = "/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/"
filt = fits.open("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_v_004_syn.fits")[1].data
wl = filt['WAVELENGTH']
trans = filt['THROUGHPUT']
err = filt['ERROR']

def plot_all_filters():
    fig = plt.figure(figsize=(11,4))
    ax = fig.add_subplot(111)
    fnames = ["U","B","V","R","I","u","g","r","i","z"]
    for name in fnames:
        wl,trans = np.load("filters/" + name + ".npy")
        ax.plot(wl,trans,label=name)
    ax.legend()
    ax.set_ylabel("Normalized transmission")
    ax.set_xlabel(r"$\lambda$ [\AA]")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylim(-1e-4,2e-3)
    fig.subplots_adjust(bottom=0.12)
    fig.savefig("plots/filter_responses.png")

#Truncate PHOENIX model wavelength to UV/optical/IR portion
ind = (w_full > 2000) & (w_full < 40000)
ww = w_full[ind]

ff = load_flux_full(5900,3.5)[ind]*2e-28
#redden spectrum
#red = ff/deredden(ww,1.5,mags=False)

def create_norm_filter_npy(filt_name, filt_path):
    print("Processing %s" % filt_name)
    #Load filter from pysynphot database
    filt = fits.open(filt_path)[1].data
    #extract filter fields from FITS file
    wl = filt['WAVELENGTH']
    trans = filt['THROUGHPUT']
    #err = filt['ERROR']

    #create filter response interpolator
    interpolator = interp1d(wl,trans,kind='cubic')

    #truncate PHOENIX wavelength vector to filter range
    filt_ind = (ww > wl[0]) & (ww < wl[-1])
    wl_filt = ww[filt_ind]
    fl_filt = ff[filt_ind]

    #interpolate filter at PHOENIX spacings
    S = interpolator(wl_filt)

    #convert response function to energy integrating (for numerical convenience, since the synthetic photometry is equivalent (Bessell 2012, A.11)).
    Si = wl_filt * S
    const = trapz(Si,wl_filt)
    S_cn = Si/const

    #write wavelength and normalized filter energy response to file
    np.save("filters/%s.npy" % filt_name, np.array([wl_filt,S_cn]))

#Synthetic photometry is then evaluated by
# trapz(fl_filt * S_cn, wl_filt)


#f_lam = trapz(fl_filt * S, wl_filt)/trapz(S,wl_filt)
#f_lam2 = trapz(fl_filt * S * wl_filt, wl_filt)/trapz(S * wl_filt, wl_filt)
#print(f_lam)
#print(f_lam2)

def write_filters():
    #[filt_name, filt_path]
    #All filters are Landolt or SLOA
    #Landolt Bands	 U B V R I (= Johnson UBV + Cousins RI) Also known as Johnson-Cousins
    filts = [["U","landolt_u_004_syn.fits"],
            ["B","landolt_b_004_syn.fits"],
            ["V","landolt_v_004_syn.fits"],
            ["R","landolt_r_004_syn.fits"],
            ["I","landolt_i_004_syn.fits"],
            ["u","sdss_u_005_syn.fits"],
            ["g","sdss_g_005_syn.fits"],
            ["r","sdss_r_005_syn.fits"],
            ["i","sdss_i_005_syn.fits"],
            ["z","sdss_z_005_syn.fits"]]
    for item in filts:
        name,path = item
        create_norm_filter_npy(name,base+path)

def main():
    #write_filters()
    plot_all_filters()


if __name__=="__main__":
    main()

'''
<f_lam> = integrate(f_lam * S * wl) / integrate(S * wl)

This is converting f_lam to proportional to photons (since S in pysynphot is indeed a photonic passband), and seeing how many photons it counts. Basically, it is taking the photon-weighted average of f_lam. 

S(λ) is the dimensionless bandpass throughput function, and the division by hν = hc / λ converts the energy flux to a photon flux as is appropriate for photon-counting detectors. (ie, this is independent of the passband function).

For speed, this means we could pre-evaluate the numerator, and scale S such that the integral becomes normalized.
'''
