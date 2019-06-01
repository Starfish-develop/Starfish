import logging
from copy import deepcopy
import warnings
from collections import OrderedDict, deque
import toml
from typing import List, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import rotational_broaden, resample, doppler_shift, extinct, rescale, chebyshev_correct
from .likelihoods import order_likelihood
from .kernels import global_covariance_matrix, local_covariance_matrix

class ChunkedModel:
    pass

class SpectrumModel:
    pass