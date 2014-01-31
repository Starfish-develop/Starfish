#testing the idea of cached interpolator

import itertools
import numpy as np

#Take example as 5010, 3.6, 0.4, 0.1
w,x,y,z = 0.1, 0.2, 0.8, 0.5



temp = (5000, 5100)
logg = (3.5, 4.0)
Z = (0.0, 0.5)
alpha = (0.0, 0.2)

ind = (0, 1)

vars = itertools.product(temp, logg, Z, alpha)
#inds = itertools.product(ind, ind, ind, ind)
weights = itertools.product((1-w, w), (1-x, x), (1-y, y), (1-z, z))

for i, var in zip(weights, vars):
    print(i, var)

#create two arrays of vars, weights and do average