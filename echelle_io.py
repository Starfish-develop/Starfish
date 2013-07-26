import pyfits as pf
import numpy as np

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

#wechelletxt()
#efile = rechelletxt()
