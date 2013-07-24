import numpy as np
import matplotlib.pyplot as plt


wl,fl = np.loadtxt("GWOri_cn/23.txt",unpack=True)
w = np.load("wave_trim.npy")

def plot_wl():
    plt.plot(wl)
    plt.xlabel("Index")
    plt.ylabel(r"Wavelength $\AA$")
    plt.savefig("plots/WAVE_GWOri_solution.png")

def plot_wl_short():
    plt.plot(w)
    plt.xlabel("Index")
    plt.ylabel(r"Wavelength $\AA$")
    plt.show()


def main():
    plot_wl()
    #plot_wl_short()
    pass

if __name__=="__main__":
    main()

