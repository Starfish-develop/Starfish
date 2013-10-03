import numpy as np
from numpy.fft import fft,ifft,fftshift,ifftshift,fftfreq
import matplotlib.pyplot as plt

__author__ = 'ian'

'''
Questions to address:

When should you do ifftshift, fftshift
Different between odd numbered and even arrays?
When should you use fftfreqs?
When should you use fft vs ifft?
When should you use windowing?
How do you do padding for sinc interpolation?

'''

