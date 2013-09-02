import numpy as np
import astropy.io.fits as pf
from astropy.io import ascii
from scipy.interpolate import interp1d, griddata, NearestNDInterpolator

wl_file = pf.open("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
w = wl_file[0].data
wl_file.close()
ind = (w > 3700.) & (w < 10000.)
w = w[ind]

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


#T_points = np.array([2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,6300,6400,6500,6600,6700,6800,6900,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000,9200,9400,9600,9800,10000,10200,10400,10600,10800,11000,11200,11400,11600,11800,12000])

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
    return f

# Interpolation routines
def interpolate_test():
    f58 = load_flux(5800,4.5)
    f60 = load_flux(6000,4.5)
    bit = np.array([5800,6000])
    f = np.array([f58,f60]).T
    func = interp1d(bit,f)
    f59 = load_flux(5900,4.5)
    f59i = func(5900)
    print(np.sqrt(np.sum((f59 - f59i)**2)))
    #plt.plot(w,f59-f59i)
    plt.plot(w,f59i)
    plt.show()

    #Now call func for any values between 5800, 5900, returns the full flux.

fluxes = flux_interpolator_np()

def main():
    #Rewrite Flux
    #for temp in Ts:
    #    for logg in loggs:
    #        rewrite_flux(temp,logg)
    #write_Ts_loggs()
    #rewrite_wl()
    #load_npy(5700,4.5)
    #load_fits(5700,4.5)
    pass


if __name__=="__main__":
    main()
