import numpy as np
import pyfits as pf

wl_file = pf.open("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
w = wl_file[0].data
wl_file.close()
ind = (w > 3700.) & (w < 10000.)
w = w[ind]

Ts = np.arange(2300,12001,100)
loggs = np.arange(0.0,6.1,0.5)

def rewrite_flux(temp,logg):
    rname="HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp=temp,logg=logg)

    wname="HiResNpy/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.npy".format(temp=temp,logg=logg)

    try:
        flux_file = pf.open(rname)
        f = flux_file[0].data
        flux_file.close()
        f = f[ind]
        print("Loaded " + rname)
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


def main():
    #for temp in Ts:
    #    for logg in loggs:
    #        rewrite_flux(temp,logg)
    rewrite_wl()
    #load_npy(5700,4.5)
    #load_fits(5700,4.5)


if __name__=="__main__":
    main()
