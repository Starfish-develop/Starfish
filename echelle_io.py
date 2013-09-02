#import pyfits as pf
from astropy.io import fits as pf
import numpy as np
from astropy.io import ascii

#bname = "GWOri_c"
#bname = "Vega_2012-04-02_12h50m48s_cb.norm.crop.spec"
#bname = "Feige34_2012-04-29_03h42m55s_cb.norm.crop.spec"
bname = "11LORI_2012-04-02_02h46m34s_cb.norm.spec"
dname = "TRES_spectra/Vega/"
bname = "11LORI_2012-04-02_02h46m34s_cb.norm.flux.spec"
#fname = dname + bname + ".fits"
#fname = "FLAT_2012-04-01_23h18m54s_all_cb.blaze.spec.fits"
fname = "FLAT_2012-04-01_23h18m54s_all_cb.blaze.spec.crop.fits"
#fname = "GWOri_nf.fits"
fname = "GWOri_cf.fits"
#odir = "GWOri_f"
odir = "GWOri_cf"
#odir = "Blaze"

#Number of orders
#f = pf.open(fname)
#norder = f[0].data.shape[0]
#f.close()
#For TRES data this will always be 51
norder = 51

def wechelletxt():
    from pyraf import iraf
    for i in range(1,norder+1):
        inp = fname + "[*,{:d}]".format(i)
        out = odir + "/{:0>2d}.txt".format(i)
        print(inp,out)
        iraf.wspectext(input=inp,output=out)

def rechelletxt(bname):
    espec = []
    for i in range(1,norder+1):
        inp = bname + "/{:0>2d}.txt".format(i)
        wl,fl = np.loadtxt(inp,unpack=True)
        espec.append([wl,fl])
    print("{:d} orders of echelle read".format(norder))
    return espec

def rechellenp(bname):
    '''Reads text files similar to rechelletxt, but instead returns a 3D numpy array that takes on the shape (norders, 2, len_wl). For example, GWOri has shape (51,2,2304). Assumes each order is the same length.'''
    inp = bname + "/01.txt"
    wl,fl = np.loadtxt(inp,unpack=True)
    len_wl = len(wl)
    spec = np.empty((norder,2,len_wl))

    for i in range(norder):
        inp = bname + "/{:0>2d}.txt".format(i+1)
        spec[i] = np.loadtxt(inp,unpack=True)
    return spec

def rechellenpflat(bname):
    '''Reads text files similar to rechelletxt, but instead returns two 2D numpy arrays of shape (norders, len_wl). The first is wl, the second fl. For example, GWOri has shape (51,2304). Assumes each order is the same length.'''
    inp = bname + "/01.txt"
    wl,fl = np.loadtxt(inp,unpack=True)
    len_wl = len(wl)
    wls = np.empty((norder,len_wl))
    fls = np.empty((norder,len_wl))

    for i in range(norder):
        inp = bname + "/{:0>2d}.txt".format(i+1)
        wl, fl = np.loadtxt(inp,unpack=True)
        wls[i] = wl
        fls[i] = fl
    return [wls,fls]


def create_sigma_file():
    blaze = rechelletxt("Blaze")
    length = len(blaze[0][0])
    order,sigma_norm = np.loadtxt("sigmas.dat",unpack=True)
    sigmas = np.zeros((51,length))
    for i in range(norder):
        wl,bl = blaze[i]
        noise = 1/np.sqrt(bl)
        norm_noise = sigma_norm[i] * noise/np.min(noise)
        sigmas[i] = norm_noise
    np.save("sigmas.npy",sigmas)
    print("Wrote sigmas.npy")
    return sigmas

#sigmas = create_sigma_file()

def load_masks():
    '''Loads all of the masking regions. Returns a list that is accesed by start, end = masking_region[order][region]'''
    data = ascii.read("masking_regions.dat")
    masking_region = []
    for i in range(1,52):
        order_list = []
        mask_arr = (i == data["Order"])
        if np.sum(mask_arr) >= 1:
            for j in data[mask_arr]:
                order_list.append([j[1],j[2]])

        masking_region.append(order_list)
    print("Loaded Masks")

    return masking_region

def make_bool_masks(fname):
    '''Takes the masks from load_masks() and turns these into boolean arrays that can be easily applied when doing chi^2 comparisons or plots.'''
    efile = rechelletxt(fname)
    masks = load_masks() #Load masking region from text file
    masks_array = np.ones((51, 2299),dtype=np.bool)
    for order in range(51):
        order_masks = masks[order]
        wl,fl = efile[order]
        len_wl = len(wl)
        if len(order_masks) >= 1:
            for mask in order_masks:
                start,end = mask
                ind = (wl > start) & (wl < end)
                masks_array[order] *= -ind #multiplication knocks out Falses

    np.save("masks_array.npy", masks_array)
    print("Created mask array")


spec = rechellenpflat("GWOri_f")

def main():
    #wechelletxt()
    #efile = rechelletxt()
    #create_sigma_file()
    #make_bool_masks("GWOri_cf")
    pass

if __name__=="__main__":
    main()
