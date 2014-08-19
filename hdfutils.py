#Base package for all HDF tools

#Basically, import this package and all of the argparsing, file commenting, should be done already.

#Scripts like hdfmultiple, hdfcat, hdfplot, and hdfregion should all use this common core.


class FlatchainTree:
    '''
    Object defined to wrap a Flatchain structure in order to facilitate combining, burning, etc.

    The Tree will always follow the same structure.

    flatchains.hdf5:

    stellar samples:    stellar

    folder for model:   0

        folder for order: 22

                        cheb
                        cov
                        cov_region00
                        cov_region01
                        cov_region02
                        ....


        folder for order: 23

                        cheb
                        cov
                        cov_region00
                        cov_region01
                        cov_region02
                        ....

    folder for model:   1


    '''
    pass

    def __add__(self, other):
        pass


class OldFlatchainTree:
    '''
    The old structure which assumed only 1 DataSpectrum. For legacy's sake.
    '''
    pass
