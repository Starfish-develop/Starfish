import logging
import sys
from nptyping import Array

import numpy as np
from scipy.linalg import cho_factor, cho_solve

log = logging.getLogger(__name__)


def order_likelihood(model: "Starfish.models.SpectrumModel") -> float:
    flux, cov = model()
    np.fill_diagonal(cov, cov.diagonal() + np.finfo(cov.dtype).eps)

    try:
        factor, flag = cho_factor(cov, overwrite_a=True)
    except np.linalg.LinAlgError:
        model.log.warning(
            "failed to decompose covariance. Entering covariance debugger"
        )
        covariance_debugger(cov)
        sys.exit()

    R = flux - model.data.fluxes
    model.residuals.append(R)
    logdet = 2 * np.sum(np.log(factor.diagonal()))
    sqmah = R @ cho_solve((factor, flag), R)
    return -(logdet + sqmah) / 2
