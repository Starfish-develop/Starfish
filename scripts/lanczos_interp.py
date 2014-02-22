import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import PHOENIX_tools as pt
from fft_interpolate import gauss_taper
from scipy.signal import resample, hann, kaiser, boxcar
from scipy.special import i0, iv
from scipy.integrate import trapz
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from sinc_interpolate import Sinc_w

#fftshift and ifftshift have different behaviours depending on whether N is odd or even. When even, they behave the
# same. The only important time to use them is when you are shifting an odd array back and forth immediately. Not if
# you're shifting back from a FFT

#fftshift(fftfreq()) is the correct one to use




w_full = pt.w_full
#Shrink to just Dave's Order
ind = (w_full > 5120) & (w_full < 5220)
f_full = pt.load_flux_full(5900, 3.5, True)[ind]
w_full = w_full[ind]

c_kms = 2.99792458e5 #km s^-1


@np.vectorize
def L(x, a=4):
    if np.abs(x) < a:
        return np.sinc(x) * np.sinc(x / a)
    else:
        return 0.


@np.vectorize
def window(x, name, a=2, alpha=5):
    if np.abs(x) <= a:
        if name == 'lanczos':
            return np.sinc(x / a)
        if name == 'hann':
            return 0.5 * (1 + np.cos(np.pi * x / a))
        if name == 'kaiser':
            return i0(np.pi * alpha * np.sqrt(1 - (x / a) ** 2)) / i0(np.pi * alpha)
        if name == None or window == 'boxcar':
            return 1.
    else:
        return 0.


def plot_windows():
    windows = ['lanczos', 'hann', 'kaiser']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = np.linspace(-2, 2)
    ax.plot(xs, window(xs, 'lanczos'), label='Lanczos')
    ax.plot(xs, window(xs, 'hann'), label='Hann')
    ax.plot(xs, window(xs, 'kaiser', alpha=2), label='Kaiser=2')
    ax.plot(xs, window(xs, 'kaiser', alpha=10), label='Kaiser=10')
    ax.legend()
    ax.set_xlabel(r"$\pm a$")
    fig.savefig("plots/windows.png")


def kaiser_discrete(alpha):
    N = 29
    M = N - 1
    ns = np.arange(M + 0.1)
    z = 2 * ns / M - 1
    wn = i0(np.pi * alpha * np.sqrt(1 - (z) ** 2)) / i0(np.pi * alpha)
    return wn


def plot_kaiser_discrete():
    plt.plot(kaiser_discrete(3), label="3")
    plt.plot(kaiser_discrete(5), label="5")
    plt.plot(kaiser_discrete(14), label="14")
    plt.show()


def sinc_w(x, name='lanczos', a=2, alpha=5):
    '''Return a windowed sinc (for interpolation) using window and scale parameter (in pixels) of a.'''
    w0 = np.sinc(x)
    return w0 * window(x, name=name, a=a, alpha=alpha)


def get_db_response(xs, name, a=2, alpha=5, n=400):
    F = fft(ifftshift(sinc_w(xs, name, a=a, alpha=alpha)))
    F = F / F[0]
    #Fs = fftshift(F)
    return 10 * np.log10(np.abs(F))


def sinc_w_interp(lam, wl, fl, name='lanczos', a=2, alpha=5):
    '''lam is the spot to interpolate to, while wl and fl are the distcrete wavelengths and fluxes.'''
    #find starting index for floor(lam)
    floor_ind = np.argwhere(lam > wl)[-1][0]
    b_i = floor_ind - a + 1
    e_i = floor_ind + a + 1 #+1 for slicing
    wls = wl[b_i:e_i]
    dlam = wl[floor_ind] - wl[floor_ind - 1]
    fls = fl[b_i:e_i]
    fl = np.sum(fls * sinc_w((lam - wls) / dlam, name, a, alpha))
    return fl

#wl = np.arange(10,20.1,0.5)
#fl = wl
#print(sinc_w_interp(15.25, wl, fl, a=5))
#print(sinc_w_interp(15.25, wl, fl, name='hann', a=10))




def plot_sinc_windows():
    fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
    xs = np.linspace(-2, 2., num=200)
    xs4 = np.linspace(-5, 5, num=500)
    y2s = sinc_w(xs, 'lanczos')
    y4s = sinc_w(xs4, 'lanczos', a=5)
    yks = sinc_w(xs4, 'kaiser', a=5, alpha=5)
    yks2 = sinc_w(xs, 'kaiser', a=2, alpha=5)
    ax[0].plot(xs, y2s, "b", label='Lanczos, a=2')
    ax[0].plot(xs4, y4s, 'g', label='Lanczos, a=5')
    ax[0].plot(xs4, yks, "r", label='Kaiser 5, a=5')
    ax[0].plot(xs, yks2, "c", label='Kaiser 5, a=2')
    #ax[0].plot(xs,sinc_w(xs, 'hann'),label='Hann')
    #ax[0].plot(xs,sinc_w(xs, 'kaiser',alpha=5),label='Kaiser=5')
    #ax[0].plot(xs,sinc_w(xs, 'kaiser',alpha=10),label='Kaiser=10')
    #xs4 = np.linspace(-4,4,num=100)
    #ax[0].plot(xs4,sinc_w(xs4, 'lanczos', a = 4), label='Lanczos,a=4')

    ax[0].legend()
    ax[0].set_xlabel(r"$\pm a$")

    #n=400 #zeropadd FFT
    #freqs2 = fftfreq(len(y2s),d=xs[1]-xs[0])
    #freqs4 =fftfreq(400,d=xs4[1]-xs4[0])
    ysh = ifftshift(y2s)
    pady = np.concatenate((ysh[:100], np.zeros((1000,)), ysh[100:]))
    freq2 = fftshift(fftfreq(len(pady), d=0.02))

    ys4h = ifftshift(y4s)

    pad4y = np.concatenate((ys4h[:250], np.zeros((2000,)), ys4h[250:]))

    freq4 = fftshift(fftfreq(len(pad4y), d=0.02))
    fpady = fft(pady)
    fpad4y = fft(pad4y)
    ax[1].plot(freq2, 10 * np.log10(np.abs(fftshift(fpady / fpady[0]))))
    ax[1].plot(freq4, 10 * np.log10(np.abs(fftshift(fpad4y / fpad4y[0]))))

    ysk = ifftshift(yks)
    padk = np.concatenate((ysk[:250], np.zeros((2000,)), ysk[250:]))
    fpadk = fft(padk)
    ax[1].plot(freq4, 10 * np.log10(np.abs(fftshift(fpadk / fpadk[0]))))
    ysk2 = ifftshift(yks2)
    padk2 = np.concatenate((ysk2[:100], np.zeros((1000,)), ysk2[100:]))
    fpadk2 = fft(padk2)
    ax[1].plot(freq2, 10 * np.log10(np.abs(fftshift(fpadk2 / fpadk2[0]))))
    #ax[1].plot(freqs4, fft(ifftshift(
    #ax[1].plot(freqs, get_db_response(xs, 'hann'),label='Hann')
    #ax[1].plot(freqs, get_db_response(xs, 'kaiser',alpha=5),label='Kaiser=5')
    #ax[1].plot(freqs, get_db_response(xs, 'kaiser',alpha=10),label='Kaiser=10')
    #ax[1].legend()
    ax[1].set_ylabel("dB")
    ax[1].set_xlabel("cycles/a")
    plt.show()
    #fig.savefig("plots/sinc_windows.png")



    ##can try with different windows and see the effect on a spectrum.

# take PHOENIX spectrum linear in lambda over some narrow wavelength, FFT, convolve with Gaussian. Then see how
# different windows recover features of the spectrum.

#Might consider Hann, Kaiser

#How can we get a handle on what is the problem -- is it spectral leakage or is it blurring out frequencies? Zoom in
# on actual FT to see?


def plot_truncation():
    n_wl = len(w_full)
    if n_wl % 2 == 0:
        print("Full Even")
    else:
        print("Full Odd")

    out = fft(ifftshift(f_full))
    w0 = ifftshift(w_full)[0]
    freqs = fftfreq(len(f_full), d=0.01) # spacing, Ang
    #sfreqs = fftshift(freqs)
    taper = gauss_taper(freqs, sigma=0.0496) #Ang, corresponds to 2.89 km/s at 5150A.
    tout = out * taper
    #ignore all samples higher than 5.0 km/s = 0.5/0.0429 cycles/Ang
    sc = 0.5 / 0.0429
    ind = np.abs(freqs) <= sc
    ttrim = tout[ind]
    if len(ttrim) % 2 == 0:
        print("Trim Even")
    else:
        print("Trim Odd")

    f_restored = fftshift(ifft(tout))
    np.save("PH6.8kms_0.01ang.npy", np.array([w_full, np.abs(f_restored)]))
    scale_factor = len(ttrim) / n_wl
    f_restored2 = scale_factor * fftshift(ifft(ttrim))
    d = freqs[1]
    print(d)
    # must keep track of which index!! AND which ifftshift vs fftshift
    raw_wl = fftshift(fftfreq(len(ttrim), d=d))
    wl_restored = raw_wl + w0
    np.save("PH2.5kms.npy", np.array([wl_restored, np.abs(f_restored2)]))

    plt.plot(w_full, f_restored)
    plt.plot(wl_restored, f_restored2, "go")
    plt.show()
    return wl_restored


def plot_FFTs():
    fig, ax = plt.subplots(nrows=4, figsize=(15, 8))

    #n = 100000
    ##Take FFT of f_grid
    #w_full = w_full[:-1]
    #f_full = f_full[:-1]
    out = np.fft.fft(np.fft.fftshift(f_full))
    freqs = fftfreq(len(f_full), d=0.01) # spacing, Ang
    sfreqs = fftshift(freqs)
    taper = gauss_taper(freqs, sigma=0.0496) #Ang, corresponds to 2.89 km/s at 5150A.
    tout = out * taper

    ax[0].plot(sfreqs, np.fft.fftshift(out))
    ax[1].plot(tout)
    #ax[1].plot(sfreqs,np.fft.fftshift(taper)*tout[0])
    ax[1].set_xlabel(r"cycles/$\lambda$")

    n_wl = len(w_full)
    print(n_wl)
    #trucate at 3 Sigma = 3.21 cycles/Ang
    #ind = np.abs(freqs) < 3.21
    #Pad tout and window
    f_restored = ifftshift(ifft(tout))
    #ax[2].plot(w_full,f_full)
    ax[2].plot(w_full, f_restored, "bo")

    #where to pad zeros? In the middle near high frequencies, so it goes +, zeros, -
    zeros = np.zeros((n_wl,))
    nyq = np.ceil(n_wl / 2) #find where the nyquist frequency is stored in the array
    t_pack = np.concatenate((tout[:nyq], zeros, tout[nyq:]))
    wl0 = w_full[nyq]

    scale_factor = len(t_pack) / n_wl
    f_restored2 = scale_factor * ifftshift(ifft(t_pack))
    wls = ifftshift(fftfreq(len(t_pack), d=0.01)) + wl0
    #ax[2].plot(wls,f_restored2)

    #print(np.sum(f_restored),np.sum(f_restored2))
    #print(trapz(f_restored,w_full),trapz(f_restored2,wls))

    #sample at an offset in phase
    half_shift = tout * np.exp(-2j * np.pi * freqs * 0.0248)
    f_restored_shift = ifftshift(ifft(half_shift))
    ax[2].plot(w_full - 0.0248, f_restored_shift, "go")

    plt.show()


def plot_pixel_effect():
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    out = fft(ifftshift(f_full))
    freqs = fftfreq(len(f_full), d=0.01) # spacing, Ang
    sfreqs = fftshift(freqs)
    taper = gauss_taper(freqs, sigma=0.0496) #Ang, corresponds to 2.89 km/s at 5150A.
    tout = out * taper

    for ax in axs[:, 0]:
        ax.plot(sfreqs, fftshift(tout) / tout[0])
        ax.plot(sfreqs, fftshift(taper))
        ax.plot(sfreqs, 0.0395 * np.sinc(0.0395 * sfreqs))
        ax.plot(sfreqs, 0.0472 * np.sinc(0.0472 * sfreqs))

    for ax in axs[:, 1]:
        ax.plot(sfreqs, 10 * np.log10(np.abs(fftshift(tout) / tout[0])))
        ax.plot(sfreqs, 10 * np.log10(np.abs(fftshift(taper))))
        ax.plot(sfreqs, 10 * np.log10(np.abs(0.0395 * np.sinc(0.0395 * sfreqs))))
        ax.plot(sfreqs, 10 * np.log10(np.abs(0.0472 * np.sinc(0.0472 * sfreqs))))

    axs[0, 0].set_ylabel("Norm amp")
    axs[1, 0].set_ylabel("Norm amp")
    axs[0, 1].set_ylabel("dB")
    axs[1, 1].set_ylabel("dB")
    for ax in axs.flatten():
        ax.set_xlabel(r"cycles/$\lambda$")
    plt.show()


def compare_interpolated_spectrum():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    out = fft(ifftshift(f_full))
    freqs = fftfreq(len(f_full), d=0.01) # spacing, Ang
    sfreqs = fftshift(freqs)
    taper = gauss_taper(freqs, sigma=0.0496) #Ang, corresponds to 2.89 km/s at 5150A.
    tout = out * taper

    ax.plot(sfreqs, fftshift(tout))

    wl_h, fl_h = np.abs(np.load("PH6.8kms_0.01ang.npy"))
    wl_l, fl_l = np.abs(np.load("PH2.5kms.npy"))

    #end edges
    wl_he = wl_h[200:-200]
    fl_he = fl_h[200:-200]
    interp = Sinc_w(wl_l, fl_l, a=5, window='kaiser')
    fl_hi = interp(wl_he)

    d = wl_he[1] - wl_he[0]
    out = fft(ifftshift(fl_hi))
    freqs = fftfreq(len(out), d=d)
    ax.plot(fftshift(freqs), fftshift(out))

    plt.show()


@np.vectorize
def lan_intp(x):
    x_floor = np.argwhere((xs < x))[-1][0] # x = 13, x_floor = 4
    b_i = x_floor - a + 1
    e_i = x_floor + a
    ii = np.arange(b_i, e_i + 1)
    s_x = np.sum(ys[ii] * L(x - xs[ii]))
    return s_x


def plot_lan_intp():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs_fine = np.linspace(14.1, 20.9)
    ax.plot(xs_fine, lan_intp(xs_fine))
    ax.plot(xs, ys, "o")
    plt.show()


def test_interpolate():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs_fine = np.linspace(10.1, 24.9)
    interp = interp1d(xs, ys)
    ax.plot(xs_fine, interp(xs_fine))
    ax.plot(xs, ys, "o")
    plt.show()

#Take 2.5kms sampled spectrum, then Lanczos interpolate to the original PHOENIX wl points
def test_lan_intp():
    wl_h, fl_h = np.abs(np.load("PH6.8kms_0.01ang.npy"))
    wl_l, fl_l = np.abs(np.load("PH2.5kms.npy"))

    #end edges
    wl_he = wl_h[200:-200]
    fl_he = fl_h[200:-200]
    interp = Sinc_w(wl_l, fl_l, a=5, window='kaiser')
    fl_hi = interp(wl_he)
    #print(np.where(np.isnan(fl_hi)==True))
    #print(len(fl_hi))
    bad_array = np.array([355, 3688, 7021])
    print(wl_he[bad_array])
    #print(interp(wl_he[bad_array]))



    fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
    ax[0].plot(wl_h, fl_h)
    ax[0].plot(wl_l, fl_l, "go")
    ax[0].plot(wl_he, fl_hi, "r.")

    ax[1].plot(wl_h, fl_h)
    ax[1].plot(wl_l, fl_l, "go")
    ax[1].plot(wl_he, fl_hi, "r.")
    plt.show()

#test_interpolate()
#plot_lan_intp()
#plot_FFTs()
#plot_windows()
#plot_kaiser_discrete()
#plot_sinc_windows()
#plot_pixel_effect()
#wl = plot_truncation()
#test_lan_intp()
compare_interpolated_spectrum()
