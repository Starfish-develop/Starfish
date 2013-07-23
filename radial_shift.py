import numpy as np


def shift_vel(line, v):
    '''Shift a line a certain velocity. Negative velocities refer to blueshift.'''
    beta = (v*1e5)/c
    return line * np.sqrt((1. + beta)/(1. - beta))
    
def gamma(v):
    '''Calculates relativistic `\gamma` based upon velocity (in km/s).
    ..math:
        
        \gamma = (1 - v_z^2/c^2)^{-1/2}'''
    #Hasn't been vetted yet
    return (1.0 - (v*1e5)**2/c**2)**(-0.5)
