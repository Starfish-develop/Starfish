#Load some spectra
import numpy as np
import bokeh
from bokeh.plotting import *
from bokeh.objects import Range1d

output_file("image.html", title="bokeh_matrix.py image")

wl = np.load("WASP14.wls.npy")[22][1000:2000]
fl = np.load("WASP14.fls.npy")[22][1000:2000]
flm = np.load("WASPfl.npy")[1000:2000]
res = np.load("WASP_resid.npy")[1000:2000]


N = len(wl)
#img = np.random.normal(size=(N,N))
img = np.load("S.npy").T[1000:2000, 1000:2000]


min_x, max_x = np.min(wl), np.max(wl)
min_y, max_y = np.min(wl), np.max(wl)
xy_range = Range1d(start=min_x, end=max_x)



hold()

figure(title="WASP-14",
    tools="pan,wheel_zoom,box_zoom,reset,previewsave,select", 
    plot_width=800, plot_height=300)

plot0 = line(wl, fl, line_width=1.5, legend="WASP14", x_range=xy_range, color="blue")
plot1 = line(wl, flm, line_width=1.5, legend="WASP14", x_range=xy_range, color="red")

figure(title="Residuals",
    tools="pan,wheel_zoom,box_zoom,reset,previewsave,select", 
    plot_width=800, plot_height=300)
#
plot1 = line(wl, fl - flm, line_width=1.5, legend="noise residuals",x_range=xy_range)

figure(title="Covariance",
    tools="pan,wheel_zoom,box_zoom,reset,previewsave,select", 
    plot_width=800, plot_height=800)

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

show()
