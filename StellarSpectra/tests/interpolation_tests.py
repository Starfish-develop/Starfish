def interpolate_raw_test_temp():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")
    wl = wls[22]
    ind2 = (m.w_full > wl[0]) & (m.w_full < wl[-1])
    w = m.w_full[ind2]
    f58 = load_flux_npy(5800, 3.5)[ind2]
    f59 = load_flux_npy(5900, 3.5)[ind2]
    f60 = load_flux_npy(6000, 3.5)[ind2]

    bit = np.array([5800, 6000])
    f = np.array([f58, f60]).T
    func = interp1d(bit, f)
    f59i = func(5900)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(w, (f59 - f59i) * 100 / f59)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/5800_5900_6000_logg3.5.png")


def interpolate_raw_test_logg():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")

    wl = wls[22]
    ind2 = (m.w_full > wl[0]) & (m.w_full < wl[-1])
    w = m.w_full[ind2]
    f3 = load_flux_npy(5900, 3.0)[ind2]
    f3_5 = load_flux_npy(5900, 3.5)[ind2]
    f4 = load_flux_npy(5900, 4.0)[ind2]

    bit = np.array([3.0, 4.0])
    f = np.array([f3, f4]).T
    func = interp1d(bit, f)
    f3_5i = func(3.5)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(w, (f3_5 - f3_5i) * 100 / f3_5)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/5900_logg3_3.5_4.png")


def interpolate_test_temp():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")

    f58 = load_flux_npy(2400, 3.5)
    f59 = load_flux_npy(2500, 3.5)
    f60 = load_flux_npy(2600, 3.5)
    bit = np.array([2400, 2600])
    f = np.array([f58, f60]).T
    func = interp1d(bit, f)
    f59i = func(2500)

    d59 = m.degrade_flux(wl, m.w_full, f59)
    d59i = m.degrade_flux(wl, m.w_full, f59i)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(wl, (d59 - d59i) * 100 / d59)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/2400_2500_2600_logg3.5_degrade.png")


def interpolate_test_logg():
    base = 'data/LkCa15//LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
    wls = np.load(base + "wls.npy")
    fls = np.load(base + "fls.npy")

    wl = wls[22]

    f3 = load_flux_npy(2400, 3.0)
    f3_5 = load_flux_npy(2500, 3.5)
    f4 = load_flux_npy(2600, 4.0)

    bit = np.array([3.0, 4.0])
    f = np.array([f3, f4]).T
    func = interp1d(bit, f)
    f3_5i = func(3.5)

    d3_5 = m.degrade_flux(wl, m.w_full, f3_5)
    d3_5i = m.degrade_flux(wl, m.w_full, f3_5i)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.axhline(0, color="k")
    ax.plot(wl, (d3_5 - d3_5i) * 100 / d3_5)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.set_ylabel("Fractional Error [\%]")
    fig.savefig("plots/interp_tests/2500logg3_3.5_4_degrade.png")
