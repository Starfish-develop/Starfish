import numpy as np
import astropy.io.fits as pf
from astropy.io import ascii
from scipy.interpolate import interp1d, griddata, NearestNDInterpolator
from echelle_io import rechellenpflat
#import model as m
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as FSF
import h5py

wl_file = pf.open("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
w_full = wl_file[0].data
wl_file.close()
ind = (w_full > 3700.) & (w_full < 10000.)
w = w_full[ind]
len_p = len(w)

Ts = np.arange(2300,12001,100)
loggs = np.arange(0.0,6.1,0.5)

def write_Ts_loggs():
    T_list = []
    logg_list = []
    for T in Ts:
        for logg in loggs:
            T_list.append(T)
            logg_list.append(logg)

    ascii.write({"T":T_list, "logg":logg_list} , "param_grid.txt", names=["T","logg"])


T_points = np.array([2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,6300,6400,6500,6600,6700,6800,6900,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000,9200,9400,9600,9800,10000,10200,10400,10600,10800,11000,11200,11400,11600,11800,12000])
logg_points = np.arange(0.0,6.1,0.5)

def point_resolver():
    '''Resolves a continous query in temp, logg to the nearest parameter combo in the PHOENIX grid. All available combinations are listed in param_grid.txt.'''
    points = np.loadtxt("param_grid_GWOri.txt")
    pr = NearestNDInterpolator(points,points) #Called as pr(5713, 3.45)
    return pr

def write_hdf5():
    '''create an hdf5 file of the PHOENIX grid. Go through each T point, if the corresponding logg exists, write it. If not, write zeros.'''
    f = h5py.File("LIB.hdf5","w")
    dset = f.create_dataset("LIB",(len(T_points),len(logg_points),len_p), dtype="f")
    for t,temp in enumerate(T_points):
        for l,logg in enumerate(logg_points):
            try:
                flux = load_flux_npy(temp,logg)
                print("Wrote %s, %s" % (temp,logg))
            except OSError:
                print("%s, %s does not exist!" % (temp,logg))
                flux = np.nan
            dset[t,l,:] = flux


def flux_interpolator():
    points = ascii.read("param_grid_GWOri.txt")
    T_list = points["T"].data
    logg_list = points["logg"].data
    fluxes = np.empty((len(T_list),len(w)))
    for i in range(len(T_list)):
        fluxes[i] = load_flux_npy(T_list[i], logg_list[i])
    flux_intp = NearestNDInterpolator(np.array([T_list, logg_list]).T, fluxes)
    return flux_intp

def flux_interpolator_np():
    points = np.loadtxt("param_grid_GWOri.txt")
    print(points)
    #T_list = points["T"].data
    #logg_list = points["logg"].data
    len_w = 716665
    fluxes = np.empty((len(points),len_w)) 
    for i in range(len(points)):
        fluxes[i] = load_flux_npy(points[i][0],points[i][1])
    flux_intp = NearestNDInterpolator(points, fluxes)
    return flux_intp

def rewrite_flux(temp,logg):
    rname="HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp=temp,logg=logg)

    wname="HiResNpy/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.npy".format(temp=temp,logg=logg)

    try:
        flux_file = pf.open(rname)
        f = flux_file[0].data
        flux_file.close()
        f = f[ind]
        print("Loaded " + rname)
        print(f.dtype)
        np.save(wname,f)
        print("Wrote " + wname)
        print()
    except OSError:
        print(rname + " does not exist!")

def rewrite_wl():
    np.save("ind.npy",ind)
    np.save("wave_trim.npy",w)

def load_flux_npy(temp,logg):
    rname="HiResNpy/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.npy".format(temp=temp,logg=logg)
    print("Loading " + rname)
    return np.load(rname)

def load_flux_fits(temp,logg):
    rname="HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp=temp,logg=logg)
    flux_file = pf.open(rname)
    f = flux_file[0].data
    flux_file.close()
    f = f[ind]
    print("Loaded " + rname)
    #Print Radius and Temperature
    return f

def load_flux_full(temp,logg):
    rname="HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp=temp,logg=logg)
    flux_file = pf.open(rname)
    f = flux_file[0].data
    flux_file.close()
    print("Loaded " + rname)
    return f

# Write into npz array

# Interpolation routines
def interpolate_raw_test_temp():
    wls, fls = rechellenpflat("GWOri_cf")
    wl = wls[22]
    ind2 = (m.w_full > wl[0]) & (m.w_full < wl[-1])
    w = m.w_full[ind2]
    f58 = load_flux_npy(5800,3.5)[ind2]
    f59 = load_flux_npy(5900,3.5)[ind2]
    f60 = load_flux_npy(6000,3.5)[ind2]

    bit = np.array([5800,6000])
    f = np.array([f58,f60]).T
    func = interp1d(bit,f)
    f59i = func(5900)

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    ax.axhline(0,color="k")
    ax.plot(w,(f59 - f59i)*100/f59)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/5800_5900_6000_logg3.5.png")

def interpolate_raw_test_logg():
    wls, fls = rechellenpflat("GWOri_cf")
    wl = wls[22]
    ind2 = (m.w_full > wl[0]) & (m.w_full < wl[-1])
    w = m.w_full[ind2]
    f3 = load_flux_npy(5900,3.0)[ind2]
    f3_5 = load_flux_npy(5900,3.5)[ind2]
    f4 = load_flux_npy(5900,4.0)[ind2]

    bit = np.array([3.0,4.0])
    f = np.array([f3,f4]).T
    func = interp1d(bit,f)
    f3_5i = func(3.5)

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    ax.axhline(0,color="k")
    ax.plot(w,(f3_5 - f3_5i)*100/f3_5)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/5900_logg3_3.5_4.png")

def interpolate_test_temp():
    wls, fls = rechellenpflat("GWOri_cf")
    wl = wls[22]

    f58 = load_flux_npy(2400,3.5)
    f59 = load_flux_npy(2500,3.5)
    f60 = load_flux_npy(2600,3.5)
    bit = np.array([2400,2600])
    f = np.array([f58,f60]).T
    func = interp1d(bit,f)
    f59i = func(2500)

    d59 = m.degrade_flux(wl,m.w_full,f59)
    d59i = m.degrade_flux(wl,m.w_full,f59i)

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    ax.axhline(0,color="k")
    ax.plot(wl,(d59 - d59i)*100/d59)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/2400_2500_2600_logg3.5_degrade.png")

def interpolate_test_logg():
    wls, fls = rechellenpflat("GWOri_cf")
    wl = wls[22]

    f3 = load_flux_npy(2400,3.0)
    f3_5 = load_flux_npy(2500,3.5)
    f4 = load_flux_npy(2600,4.0)

    bit = np.array([3.0,4.0])
    f = np.array([f3,f4]).T
    func = interp1d(bit,f)
    f3_5i = func(3.5)

    d3_5 = m.degrade_flux(wl,m.w_full,f3_5)
    d3_5i = m.degrade_flux(wl,m.w_full,f3_5i)

    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    ax.axhline(0,color="k")
    ax.plot(wl,(d3_5 - d3_5i)*100/d3_5)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/2500logg3_3.5_4_degrade.png")

pr = point_resolver()


def main():
    #Rewrite Flux
    #for temp in Ts:
    #    for logg in loggs:
    #        rewrite_flux(temp,logg)
    #write_Ts_loggs()
    #rewrite_wl()
    #load_npy(5700,4.5)
    #load_fits(5700,4.5)
    #interpolate_test_temp()
    #interpolate_test_logg()
    write_hdf5()
    pass


if __name__=="__main__":
    main()
