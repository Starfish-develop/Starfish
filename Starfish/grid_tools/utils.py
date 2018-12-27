import multiprocessing as mp

import numpy as np


def chunk_list(mylist, n=mp.cpu_count()):
    """
    Divide a lengthy parameter list into chunks for parallel processing and
    backfill if necessary.

    :param mylist: a lengthy list of parameter combinations
    :type mylist: 1-D list
    :param n: number of chunks to divide list into. Default is ``mp.cpu_count()``
    :type n: integer

    :returns: **chunks** (*2-D list* of shape (n, -1)) a list of chunked parameter lists.

    """
    if isinstance(mylist, np.ndarray):
        mylist = list(mylist)
    length = len(mylist)
    size = int(length / n)
    chunks = [mylist[0 + size * i: size * (i + 1)] for i in range(n)]  # fill with evenly divisible
    leftover = length - size * n
    edge = size * n
    for i in range(leftover):  # backfill each with the last item
        chunks[i % n].append(mylist[edge + i])
    return chunks


def determine_chunk_log(wl, wl_min, wl_max):
    """
    Take in a wavelength array and then, given two minimum bounds, determine
    the boolean indices that will allow us to truncate this grid to near the
    requested bounds while forcing the wl length to be a power of 2.

    :param wl: wavelength array
    :type wl: np.ndarray
    :param wl_min: minimum required wavelength
    :type wl_min: float
    :param wl_max: maximum required wavelength
    :type wl_max: float

    :returns: numpy.ndarray boolean array

    """

    # wl_min and wl_max must of course be within the bounds of wl
    assert wl_min >= np.min(wl) and wl_max <= np.max(
        wl), "determine_chunk_log: wl_min {:.2f} and wl_max {:.2f} are not within the bounds of the grid {:.2f} to {:.2f}.".format(
        wl_min, wl_max, np.min(wl), np.max(wl))

    # Find the smallest length synthetic spectrum that is a power of 2 in length
    # and longer than the number of points contained between wl_min and wl_max
    len_wl = len(wl)
    npoints = np.sum((wl >= wl_min) & (wl <= wl_max))
    chunk = len_wl
    inds = (0, chunk)

    # This loop will exit with chunk being the smallest power of 2 that is
    # larger than npoints
    while chunk > npoints:
        if chunk / 2 > npoints:
            chunk = chunk // 2
        else:
            break

    assert type(chunk) == np.int, "Chunk is not an integer!. Chunk is {}".format(chunk)

    if chunk < len_wl:
        # Now that we have determined the length of the chunk of the synthetic
        # spectrum, determine indices that straddle the data spectrum.

        # Find the index that corresponds to the wl at the center of the data spectrum
        center_wl = (wl_min + wl_max) / 2.
        center_ind = (np.abs(wl - center_wl)).argmin()

        # Take a chunk that straddles either side.
        inds = (center_ind - chunk // 2, center_ind + chunk // 2)

        ind = (np.arange(len_wl) >= inds[0]) & (np.arange(len_wl) < inds[1])
    else:
        print("keeping grid as is")
        ind = np.ones_like(wl, dtype='bool')

    assert (min(wl[ind]) <= wl_min) and (max(wl[ind]) >= wl_max), "Model" \
                                                                  "Interpolator chunking ({:.2f}, {:.2f}) didn't encapsulate full" \
                                                                  " wl range ({:.2f}, {:.2f}).".format(min(wl[ind]),
                                                                                                       max(wl[ind]),
                                                                                                       wl_min, wl_max)

    return ind


def vacuum_to_air(wl):
    """
    Converts vacuum wavelengths to air wavelengths using the Ciddor 1996 formula.

    :param wl: input vacuum wavelengths
    :type wl: numpy.ndarray

    :returns: numpy.ndarray

    .. note::

        CA Prieto recommends this as more accurate than the IAU standard.

    """
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)

    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f


def calculate_n(wl):
    """
    Calculate *n*, the refractive index of light at a given wavelength.

    :param wl: input wavelength (in vacuum)
    :type wl: np.array

    :return: numpy.ndarray
    """
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)

    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    new_wl = wl / f
    n = wl / new_wl
    print(n)


def vacuum_to_air_SLOAN(wl):
    """
    Converts vacuum wavelengths to air wavelengths using the outdated SLOAN definition.
    From the SLOAN website:

    AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)

    :param wl:
        The input wavelengths to convert


    """
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)
    air = wl / (1.0 + 2.735182E-4 + 131.4182 / wl ** 2 + 2.76249E8 / wl ** 4)
    return air


def air_to_vacuum(wl):
    """
    Convert air wavelengths to vacuum wavelengths.

    :param wl: input air wavelegths
    :type wl: np.array

    :return: numpy.ndarray

    .. warning::

        It is generally not recommended to do this, as the function is imprecise.
    """
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)

    sigma = 1e4 / wl
    vac = wl + wl * (6.4328e-5 + 2.94981e-2 / (146 - sigma ** 2) + 2.5540e-4 / (41 - sigma ** 2))
    return vac


@np.vectorize
def idl_float(idl_num):
    """
    idl_float(idl_num)
    Convert an idl *string* number in scientific notation it to a float.

    :param idl_num: the idl number in sci_notation
    :type idl_num: str

    :returns: float

    Example usage:

    .. code-block:: python

        idl_num = "1.5D4"
        idl_float(idl_num)
        15000.0 # Numpy float64

    """
    idl_num = idl_num.lower()
    # replace 'D' with 'E', convert to float
    return np.float(idl_num.replace("d", "e"))
