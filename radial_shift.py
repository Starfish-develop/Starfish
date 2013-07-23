import numpy as np



@np.vectorize
def shift_vz(lam_source, v):
    '''Given the source wavelength, lam_sounce, return the observed wavelength based upon a velocity v in km/s. Negative velocities are towards the observer (blueshift).'''
    lam_observe = lam_source * np.sqrt((c_kms + v)/(c_kms - v))
    return lam_observe
