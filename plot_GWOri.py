import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt

GW_file = pf.open("GWOri_cnm.fits")

f = GW_file[0].data

disp = 0.032920821413025
w0 = 3850.3823242188

w = np.arange(len(f)) * disp + w0

def plot_GWOri():
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.plot(w,f)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.set_xlim(3800,9100)
    plt.show()
    pass


def main():
    plot_GWOri()
    pass

if __name__=="__main__":
    main()


