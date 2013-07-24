import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

wl,fl = np.loadtxt("GWOri_cn/23.txt",unpack=True)

w = np.load("wave_trim.npy")

#Limit huge file to necessary range
ind = (w > (wl[0] - 10.)) & (w < (wl[-1] + 10))
w = w[ind]

def load_flux(temp,logg):
    fname="HiResNpy/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5.0f}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.npy".format(temp=temp,logg=logg)
    print("Loaded " + fname)
    f = np.load(fname)
    return f[ind]

def plot_grid():
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    f59_58 = load_flux(5900,4.5) - load_flux(5800,4.5)
    #ax.plot(w,f58_57)
    #ax.plot(w,f59_58)
    ax.plot(w,f59_58/f58_57)
    plt.show()

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


def main():
    #plot_grid()
    interpolate_test()
    pass


if __name__=="__main__":
    main()
    
