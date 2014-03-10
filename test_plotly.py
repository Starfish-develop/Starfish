import plotly
py = plotly.plotly("iancze", "0ttojbuvyj")
import StellarSpectra
from StellarSpectra import spectrum
from StellarSpectra import constants as C
from spectrum import DataSpectrum
import numpy as np
import sys
from astropy.io import ascii

myspec = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux",
                           orders=np.array([22]))

#Shift wl as close to 0.
vz = -15
myspec.wls = myspec.wls * np.sqrt((C.c_kms + vz) / (C.c_kms - vz))


def return_line_labels(wl, tol=1):
    '''Given a wl array, return the nearest n line labels next to the line, that are within
    tolerance = 1 Ang of each point.'''

    #for linelist_air.dat, col_starts=[3, 20], col_ends=[17, 28]
    #for linelist_kurucz.dat, col_starts=[3, 13], col_ends=[10, 20]

    lines = ascii.read("linelist_kurucz.dat", Reader=ascii.FixedWidth, col_starts=[3, 13], col_ends=[10, 20],
                       converters={'line': [ascii.convert_numpy(np.float)],
                                   'element': [ascii.convert_numpy(np.str)]}, guess=False)
    lines['line'] = 10 * lines['line'] #Convert from nanometers to AA

    #truncate list to speed execution
    ind = (lines['line'] >= np.min(wl) - tol) & (lines['line'] <= np.max(wl) + tol)
    lines = lines[ind]

    #for each wl, query all known lines that are within tol, add these to the set of known lines
    line_labels = []
    for w in wl:
        #Find nearby wl lines within tol
        ind = (w - tol <= lines['line']) & (lines['line'] <= w + tol)

        #Truncated lines
        lines_trunc = lines[ind]

        #Sort them by closeness to current pixel
        distances = np.abs(w - lines_trunc['line'])
        distance_ind = np.argsort(distances)

        #Sort lines by closest label
        lines_sort = lines_trunc[distance_ind]

        #Take only 6 lines
        lines_clip = lines_sort[:6]

        #Create a new set
        labels = "\n".join(["{} {:.2f}".format(label,line) for line, label in lines_clip])

        line_labels.append(labels)

    return line_labels

line_list = return_line_labels(myspec.wls[0], tol=0.3)

data = {'name': 'WASP-14',
            'x': myspec.wls[0],
            'y': myspec.fls[0],
            'text': line_list,
            'type': 'scatter',
            'mode': 'lines+markers'
            }


layout = {
    'xaxis': {'title': 'Wavelength (AA)'},
    'yaxis': {'title': 'Flux (ergs/s/AA/cm^2)'},
    'title': 'WASP-14'
}

response = py.plot(data, layout=layout, filename='Spectra/WASP-14', fileopt='overwrite', world_readable=True)
url = response['url']
