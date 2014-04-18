#Load some spectra
import numpy as np
import bokeh
from bokeh.plotting import *
from bokeh.objects import Range1d

output_file("spectrum.html", title="try_bokeh.py spectrum")

wl = np.load("WASP14.wls.npy")[22]
fln = np.load("fakeWASP.clean.fl.npy")
flc = np.load("fakeWASP.noisey.fl.npy")

xr = Range1d(start=np.min(wl), end=np.max(wl))

figure(title="WASP-14",
    tools="pan,wheel_zoom,box_zoom,reset,previewsave,select", 
    plot_width=800, plot_height=300)

plot0 = line(wl, fln, line_width=1.5, legend="WASP14", x_range=xr)

figure(title="Residuals",
    tools="pan,wheel_zoom,box_zoom,reset,previewsave,select", 
    plot_width=800, plot_height=300)
#
plot1 = line(wl, fln - flc, line_width=1.5, legend="noise residuals",
        x_range=xr)
#
show() 
