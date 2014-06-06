#!/usr/bin/env python

# from StellarSpectra.model import Model
# from StellarSpectra.spectrum import DataSpectrum
# from StellarSpectra.grid_tools import TRES, HDF5Interface
# import StellarSpectra.constants as C
import numpy as np
# import yaml
# import json

import plotly.plotly as py
from plotly.graph_objs import *

dspec = Scatter( #data spectrum
    x=[0, 1, 2],
    y=[10, 11, 12]
)
mspec = Scatter( #model spectrum
    x=[0, 1, 2],
    y=[11, 12, 13],
)
rspec = Scatter( #residual spectrum
    x=[0, 1, 2],
    y=[1000, 1100, 1200],
    yaxis='y2'
)
data = Data([dspec, mspec, rspec])
layout = Layout(
    yaxis=YAxis(
        domain=[0, 0.33]
    ),
    yaxis2=YAxis(
        domain=[0.33, 0.66]
    ),
)
fig = Figure(data=data, layout=layout)

plot_url = py.plot(fig, filename='Output')

#
# data = {'name': 'WASP-14',
#             'x': myspec.wls[0],
#             'y': myspec.fls[0],
#             'text': line_list,
#             'type': 'scatter',
#             'mode': 'lines+markers'
#             }
#
#
# layout = {
#     'xaxis': {'title': 'Wavelength (AA)'},
#     'yaxis': {'title': 'Flux (ergs/s/AA/cm^2)'},
#     'title': 'WASP-14'
# }
#
# response = py.plot(data, layout=layout, filename='Spectra/WASP-14', fileopt='overwrite', world_readable=True)
# url = response['url']
