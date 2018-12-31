import h5py
import numpy as np
from numpy.polynomial import Chebyshev as Ch


class DataSpectrum:
    """
    Object to manipulate the data spectrum.

    :param wls: wavelength (in AA)
    :type wls: 1D or 2D np.array
    :param fls: flux (in f_lam)
    :type fls: 1D or 2D np.array
    :param sigmas: Poisson noise (in f_lam). If not specified, will be unitary. Default is None
    :type sigmas: 1D or 2D np.array
    :param masks: Mask to blot out bad pixels or emission regions. Default is None
    :type masks: 1D or 2D np.array of boolean values

    If the wl, fl, are provided as 1D arrays (say for a single order), they will be converted to 2D arrays with length 1
    in the 0-axis.

    .. note::

       For now, the DataSpectrum wls, fls, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    """

    def __init__(self, wls, fls, sigmas=None, masks=None, orders='all', name=None):
        self.wls = np.atleast_2d(wls)
        self.fls = np.atleast_2d(fls)
        if sigmas is not None:
            self.sigmas = np.atleast_2d(sigmas)
        else:
            self.sigmas = np.ones_like(self.fls)

        if masks is not None:
            self.masks = np.atleast_2d(masks)
        else:
            self.masks = np.ones_like(self.wls, dtype='b')

        self.shape = self.wls.shape
        assert self.fls.shape == self.shape, "flux array incompatible shape."
        assert self.sigmas.shape == self.shape, "sigma array incompatible shape."
        assert self.masks.shape == self.shape, "mask array incompatible shape."

        if orders != 'all':
            # can either be a numpy array or a list
            orders = np.array(orders)  # just to make sure
            self.wls = self.wls[orders]
            self.fls = self.fls[orders]
            self.sigmas = self.sigmas[orders]
            self.masks = self.masks[orders]
            self.shape = self.wls.shape
            self.orders = orders
        else:
            self.orders = np.arange(self.shape[0])

        self.name = name

    @classmethod
    def open(cls, file, orders='all'):
        """
        Load a spectrum from a directory link pointing to HDF5 output from EchelleTools processing.

        :param base_file: HDF5 file containing files on disk.
        :type base_file: string
        :returns: DataSpectrum
        :param orders: Which orders should we be fitting?
        :type orders: np.array of indexes

        """
        # Open the HDF5 file, try to load each of these values.
        with h5py.File(file, "r") as hdf5:
            wls = hdf5["wls"][:]
            fls = hdf5["fls"][:]
            sigmas = hdf5["sigmas"][:]

            try:
                # Try to see if masks is available, otherwise return an all-true mask.
                masks = np.array(hdf5["masks"][:], dtype="bool")
            except KeyError as e:
                masks = np.ones_like(wls, dtype="bool")

        # Although the actual fluxes and errors may be reasonably stored as float32, we need to do all of the calculations
        # in float64, and so we convert here.
        # The wls must be stored as float64, because of precise velocity issues.
        return cls(wls.astype(np.float64), fls.astype(np.float64), sigmas.astype(np.float64), masks, orders, name=file)

    def write(self, filename):
        """
        Takes the current DataSpectrum and writes it to an HDF5 file.

        :param filename: The filename to write to. Will not create any missing directories.
        :type filename: str or path-like
        """

        with h5py.File(filename, 'w') as base:
            base.create_dataset('wls', data=self.wls, compression=9)
            base.create_dataset('fls', data=self.fls, compression=9)
            base.create_dataset('sigmas', data=self.sigmas, compression=9)
            base.create_dataset('masks', data=self.masks, compression=9)


    @classmethod
    def open_npy(cls, base_file, orders='all'):
        """
        Load a spectrum from a directory link pointing to .npy output from EchelleTools processing.

        :param base_file: base path name to be appended with ".wls.npy", ".fls.npy", ".sigmas.npy", and ".masks.npy" to load files from disk.
        :type base_file: string
        :returns: DataSpectrum
        :param orders: Which orders should we be fitting?
        :type orders: np.array of indexes

        """
        wls = np.load(base_file + ".wls.npy")
        fls = np.load(base_file + ".fls.npy")
        sigmas = np.load(base_file + ".sigmas.npy")
        masks = np.load(base_file + ".masks.npy")
        return cls(wls, fls, sigmas, masks, orders)

    def add_mask(self, new_mask):
        """
        Given a mask with the same self.shape, update self.masks to include the union with this new mask.
        """
        assert new_mask.shape == self.shape, "new_mask shape ({}) must be the same shape as spectrum ({}).".format(
            new_mask.shape, self.shape)

        self.masks = self.masks & new_mask

    def __str__(self):
        return "DataSpectrum object {} with shape {}".format(self.name, self.shape)


class Mask:
    """
    Mask to apply to DataSpectrum
    """

    def __init__(self, masks, orders='all'):
        assert isinstance(masks, np.ndarray), "masks must be a numpy array"
        self.masks = np.atleast_2d(masks)

        if orders != 'all':
            # can either be a numpy array or a list
            orders = np.array(orders)  # just to make sure
            self.masks = self.masks[orders]
            self.orders = orders
        else:
            self.orders = np.arange(self.masks.shape[0])

    @classmethod
    def open(cls, file, orders='all'):
        """
        Load a Mask from a directory link pointing to HDF5 file output from EchelleTools or Generate_mask.ipynb
        processing.

        :param file: HDF5 file containing files on disk.
        :type file: string
        :returns: DataSpectrum
        :param orders: Which orders should we be fitting?
        :type orders: np.array of indexes

        """
        import h5py
        with h5py.File(file, "r") as hdf5:
            masks = np.array(hdf5["masks"][:], dtype="bool")

        return cls(masks, orders)


class ChebyshevSpectrum:
    """
    A DataSpectrum-like object which multiplies downsampled fls to account for imperfect flux calibration issues.

    :param DataSpectrum: take shape from.
    :type DataSpectrum: :obj:`DataSpectrum` object

    If DataSpectrum.norders == 1, then only c1, c2, and c3 are required. Otherwise c0 is also reqired for each order.
    """

    def __init__(self, DataSpectrum, index, npoly=4):
        self.wl = DataSpectrum.wls[index]
        len_wl = len(self.wl)

        self.fix_c0 = True if index == (len(DataSpectrum.wls) - 1) else False  # Fix the last c0

        xs = np.arange(len_wl)

        # Create Ch1, etc... for each coefficient in npoly excepting logc0
        # Evaluate these and stuff them into self.T
        coeff = [1]
        T = []
        for i in range(1, npoly):
            # print("i = ", i)
            coeff = [0] + coeff
            Chtemp = Ch(coeff, domain=[0, len_wl - 1])
            Ttemp = Chtemp(xs)
            T += [Ttemp]

        # Ch1 = Ch([0, 1], domain=[0, len_wl - 1])
        # T1 = Ch1(xs)
        # Ch2 = Ch([0, 0, 1], domain=[0, len_wl - 1])
        # T2 = Ch2(xs)
        # Ch3 = Ch([0, 0, 0, 1], domain=[0, len_wl - 1])
        # T3 = Ch3(xs)

        # self.T = np.array([T1, T2, T3])
        self.T = np.array(T)
        self.npoly = npoly
        # assert self.npoly == 4, "Only handling order 4 Chebyshev for now."

        # Dummy holders for a flat spectrum
        self.k = np.ones(len_wl)
        self.k_last = self.k
        # self.c0s = np.ones(self.norders)
        # self.cns = np.zeros((self.norders, self.npoly - 1))
        # self.TT = np.einsum("in,jn->ijn", T, T)

        ##Priors
        ##    mu = np.array([0, 0, 0])
        ##    D = sigmac ** (-2) * np.eye(3)
        ##    Dmu = np.einsum("ij,j->j", D, mu)
        ##    muDmu = np.einsum("j,j->", mu, Dmu)

    def update(self, p):
        """
        Given a dictionary of coefs, create a k array to multiply against model fls

        :param p: array of coefficients
        :type p: 1D np.array
        """

        # Fix the last order c0 to 1.
        if self.fix_c0:
            c0 = 1.0
            cns = p
        else:
            c0 = 10 ** p[0]
            cns = p[1:]

        # now create polynomials for each order, and multiply through fls
        # print("T shape", self.T.shape)
        # print("cns shape", cns.shape)

        Tc = np.dot(self.T.T, cns)  # self.T.T is the transpose of self.T
        # print("Tc shape", Tc.shape)
        k = c0 * (1 + Tc)
        self.k_last = self.k
        self.k = k

    def revert(self):
        self.k = self.k_last
