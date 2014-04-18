#Load some spectra
import numpy as np
import bokeh
from bokeh.plotting import *
from bokeh.objects import Range1d

output_file("image.html", title="bokeh_matrix.py image")

wl = np.load("WASP14.wls.npy")[22][:1000]
fln = np.load("fakeWASP.clean.fl.npy")[:1000]
flc = np.load("fakeWASP.noisey.fl.npy")[:1000]

N = len(wl)
img = np.random.normal(size=(N,N))

min_x, max_x = np.min(wl), np.max(wl)
min_y, max_y = np.min(wl), np.max(wl)
xy_range = Range1d(start=min_x, end=max_x)

image(image=[img],
      x=[min_x],
      y=[min_y],
      dw=[max_x-min_x],
      dh=[max_y-min_y],
      palette=["Spectral-11"],
      x_range = xy_range,
      y_range = xy_range,
      title="Covariance",
      tools="pan,wheel_zoom,box_zoom,reset,previewsave",
      plot_width=800,
      plot_height=800
)


figure(title="WASP-14",
    tools="pan,wheel_zoom,box_zoom,reset,previewsave,select", 
    plot_width=800, plot_height=300)

plot0 = line(wl, fln, line_width=1.5, legend="WASP14", x_range=xy_range)

show()
