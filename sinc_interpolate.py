#!usr/bin/env python

import numpy as np
from scipy.signal import hann, kaiser, boxcar
from scipy.special import i0

'''Class to implement windowed (or not!) sinc interpolation. Options include Lanczos, Hann, and Kaiser. Possibly more
. Copyright Ian Czekala 2013.

Assumptions: sinc interpolation is useful for *exactly* recreating a band-limited function which has been discretely
sampled at *regular* intervals. However, true sinc interpolation requires an infinite series of points, and therefore
we must window the function. Also, we can relax the requirement that the samples need to be equally spaced at all
points and assume that as long as the samples can be treated as equally spaced within the windowed kernel,
then we should be fine.

Also keep in mind that as we increase a, the ringing phenomena decreases, but the computational time increases.'''


class Sinc_w:
    def __init__(self, x, y, window='lanczos', a=2, alpha=5):
        '''New points can be an array or single value. Location to interpolate to.
        Attributes:
            x (ndarray): the points 
            y (ndarray): the values
            name (str): the windowing function
            a (int): scale parameter specifying number of adjacent points
            window (func): the windowing function
            alpha (float): kaiser scale parameter. Has no effect on other windows.'''

        self.x = x.copy() #make internal copies
        self.y = y.copy()

        #TODO: check x and y are same length

        self.window_name = window
        self.a = a
        self.alpha = alpha
        self.window_fn = self.choose_window(self.window_name)

        #Determine valid ranges of interpolation, based on self.a
        self.x_min = x[a]
        self.x_max = x[-(a + 1)]

    def choose_window(self, name):
        '''Need to make sure these windows end at +- a'''
        if name == 'lanczos':
            return lambda x: np.sinc(x / self.a)
        if name == 'hann':
            return lambda x: 0.5 * (1 + np.cos(np.pi * x / self.a))
        if name == 'kaiser':
            return lambda x: i0(np.pi * self.alpha * np.sqrt(1. - (x / self.a) ** 2)) / i0(np.pi * self.alpha)
        if name == None or window == 'boxcar':
            return lambda x: 1.
        else:
            raise NotImplementedError("Specified window %s not implemented" % (name,))

    def set_window(self, func):
        raise NotImplementedError

    def sinc_w(self, x):
        w0 = np.sinc(x)
        return w0 * self.window_fn(x)

    def __call__(self, x):
        '''Interpolate to new points x. Need not necessarily be evenly spaced.'''

        x = np.atleast_1d(x)
        #Now do all operations as if assuming array

        if (x < self.x_min).any() or (x > self.x_max).any():
            raise ValueError("Out of interpolation bounds [%f, %f] for a=%f" % (self.x_min, self.x_max, self.a))

        #Floor index closest to value
        true_arr = x[:, np.newaxis] > self.x
        print(true_arr)
        floor_inds = np.sum(true_arr, axis=1) - 1 #dangerous because assumes in self.x in ascending order
        print(floor_inds)
        print(self.x[floor_inds])

        aarray = np.arange(-self.a + 1, self.a + 1, 1)
        inds = floor_inds[:, np.newaxis] + aarray
        print(inds)

        xs = self.x[inds]
        print(xs)
        ys = self.y[inds]
        dx = xs[:, 1] - xs[:, 0]
        sinc_pts = (x[:, np.newaxis] - xs) / dx[:, np.newaxis]
        print(sinc_pts)
        s_ind = (np.abs(sinc_pts) >= self.a)
        print(s_ind)
        sinc_pts[s_ind] = self.a
        ys[s_ind] = 0
        print(np.sum(s_ind), "on the edge or outside of filter")
        yis = np.sum(ys * self.sinc_w(sinc_pts), axis=1)
        return yis

    def __str__(self):
        return "Bounds [%f,%f]" % (self.x_min, self.x_max)


def main():
    '''Some tests'''
    xs = np.arange(10, 50.1, 0.2)
    ys = xs
    intp = Sinc_w(xs, ys, a=5, window='kaiser')
    print(intp)
    print(intp(26.5))
    print(intp(np.array([20.3, 21.6, 27.3])))
    pass


if __name__ == "__main__":
    main()



