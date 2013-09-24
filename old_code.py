def downsample(w_m,f_m,w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to exactly match the TRES wavelength bins. '''
    spec_interp = interp1d(w_m,f_m,kind="linear")

    @np.vectorize
    def avg_bin(bin0,bin1):
        mdl_ind = (w_m > bin0) & (w_m < bin1)
        wave = np.empty((np.sum(mdl_ind)+2,))
        flux = np.empty((np.sum(mdl_ind)+2,))
        wave[0] = bin0
        wave[-1] = bin1
        flux[0] = spec_interp(bin0)
        flux[-1] = spec_interp(bin1)
        wave[1:-1] = w_m[mdl_ind]
        flux[1:-1] = f_m[mdl_ind]
        return trapz(flux,wave)/(bin1-bin0)

    #Determine the bin edges
    edges = np.empty((len(w_TRES)+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    samp = avg_bin(b0s,b1s)
    return(samp)

def downsample2(w_m,f_m,w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to exactly match the TRES wavelength bins. Try this without calling the interpolation routine.'''

    @np.vectorize
    def avg_bin(bin0,bin1):
        mdl_ind = (w_m > bin0) & (w_m < bin1)
        length = np.sum(mdl_ind)+2
        wave = np.empty((length,))
        flux = np.empty((length,))
        wave[0] = bin0
        wave[-1] = bin1
        wave[1:-1] = w_m[mdl_ind]
        flux[1:-1] = f_m[mdl_ind]
        flux[0] = flux[1]
        flux[-1] = flux[-2]
        return trapz(flux,wave)/(bin1-bin0)

    #Determine the bin edges
    edges = np.empty((len(w_TRES)+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    return avg_bin(b0s,b1s)

def downsample3(w_m,f_m,w_TRES):
    '''Given a model wavelength and flux (w_m, f_m) and the instrument wavelength (w_TRES), downsample the model to exactly match the TRES wavelength bins. Try this only by averaging.'''

    #More time could be saved by splitting up the original array into averageable chunks.

    @np.vectorize
    def avg_bin(bin0,bin1):
        return np.average(f_m[(w_m > bin0) & (w_m < bin1)])

    #Determine the bin edges
    edges = np.empty((len(w_TRES)+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]
    b0s = edges[:-1]
    b1s = edges[1:]

    return avg_bin(b0s,b1s)

def downsample4(w_m,f_m,w_TRES):

    out_flux = np.zeros_like(w_TRES)
    len_mod = len(w_m)

    #Determine the bin edges
    len_TRES = len(w_TRES)
    edges = np.empty((len_TRES+1,))
    difs = np.diff(w_TRES)/2.
    edges[1:-1] = w_TRES[:-1] + difs
    edges[0] = w_TRES[0] - difs[0]
    edges[-1] = w_TRES[-1] + difs[-1]

    i_start = np.argwhere((w_m > edges[0]))[0][0] #return the first starting index for the model wavelength array

    edges_i = 1
    for i in range(len(w_m)):
        if w_m[i] > edges[edges_i]:
            i_finish = i - 1
            out_flux[edges_i - 1] = np.mean(f_m[i_start:i_finish])
            edges_i += 1
            i_start = i_finish
            if edges_i > len_TRES:
                break
    return out_flux


#Keep out here so memory keeps getting overwritten
fluxes = np.empty((4, len(wave_grid)))
def flux_interpolator_mini(temp, logg):
    '''Load flux in a memory-nice manner. lnprob will already check that we are within temp = 2300 - 12000 and logg = 0.0 - 6.0, so we do not need to check that here.'''
    #Determine T plus and minus 
    #If the previous check by lnprob was correct, these should always have elements
    #Determine logg plus and minus
    i_Tm = np.argwhere(temp >= T_points)[-1][0]
    Tm = T_points[i_Tm]
    i_Tp = np.argwhere(temp < T_points)[0][0]
    Tp = T_points[i_Tp]
    i_lm = np.argwhere(logg >= logg_points)[-1][0]
    lm = logg_points[i_lm]
    i_lp = np.argwhere(logg < logg_points)[0][0]
    lp = logg_points[i_lp]

    indexes =[(i_Tm,i_lm),(i_Tm,i_lp),(i_Tp,i_lm),(i_Tp,i_lp)]
    points = np.array([(Tm,lm),(Tm,lp),(Tp,lm),(Tp,lp)])
    for i in range(4):
    #Load spectra for these points
        #print(indexes[i])
        fluxes[i] = LIB[indexes[i]]
    if np.isnan(fluxes).any():
    #If outside the defined grid (demarcated in the hdf5 object by nan's) just return 0s
        return zero_flux

    #Interpolate spectra with LinearNDInterpolator
    flux_intp = LinearNDInterpolator(points,fluxes)
    new_flux = flux_intp(temp,logg)
    return new_flux

def flux_interpolator():
    #points = np.loadtxt("param_grid_GWOri.txt")
    points = np.loadtxt("param_grid_interp_test.txt")
    #TODO: make this dynamic, specify param_grid dynamically too
    len_w = 716665
    fluxes = np.empty((len(points),len_w)) 
    for i in range(len(points)):
        fluxes[i] = load_flux(points[i][0],points[i][1])
    #flux_intp = NearestNDInterpolator(points, fluxes)
    flux_intp = LinearNDInterpolator(points, fluxes,fill_value=1.)
    del fluxes
    print("Loaded flux_interpolator")
    return flux_intp
