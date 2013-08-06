import pyfits as pf
import numpy as np
import asciitable

bname = "GWOri_c"
fname = bname + ".fits"

#Number of orders
GW = pf.open(fname)
norder = GW[0].data.shape[0]
GW.close()

def wechelletxt():
    from pyraf import iraf
    for i in range(1,norder+1):
        inp = fname + "[*,{:d}]".format(i)
        out = bname + "/{:0>2d}.txt".format(i)
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

def create_sigma_file():
    efile = rechelletxt("GWOri_c")
    efile_n = rechelletxt("GWOri_cn")
    length = len(efile[0][0])
    order,sigma_norm = np.loadtxt("sigmas.dat",unpack=True)
    sigmas = np.zeros((51,length))
    for i in range(norder):
        fl_r = efile[i][1]
        fl = efile_n[i][1]
        #Need to replace 0.0s with next lowest value, or some arbitrarily small value.
        nanmask_r = (fl_r == 0.0)
        nanmask = (fl == 0.0)
        fl_r[nanmask_r] = 0.04
        fl[nanmask] = 0.003
        cont = fl_r/fl
        nanmask_cont = np.isnan(cont)
        print(np.sum(nanmask_r),np.min(fl_r),np.max(fl_r),np.sum(nanmask),np.min(fl),np.max(fl),np.sum(nanmask_cont))
        noise = 1/np.sqrt(cont)
        norm_noise = sigma_norm[i] * noise/np.min(noise)
        sigmas[i] = norm_noise
    np.save("sigmas.npy",sigmas)
    print("Wrote sigmas.npy")
    return sigmas

#sigmas = create_sigma_file()

def load_masks():
    '''Loads all of the masking regions. Returns a list that is accesed by start, end = masking_region[order][region]'''
    data = asciitable.read("masking_regions.dat")
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



def main():
    #wechelletxt()
    #efile = rechelletxt()
    #create_sigma_file()
    pass

if __name__=="__main__":
    main()
