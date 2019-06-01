import copy
import multiprocessing as mp
from typing import List, Union, Sequence, Callable, Optional

from flatdict import FlatterDict
import numpy as np

from Starfish import Spectrum
from .likelihoods import order_likelihood
from .order import OrderModel


class SpectrumModel:
    def __init__(
        self,
        emulator: Union[str, "Starfish.Emulator"],
        data: Union[str, Spectrum],
        grid_params: Sequence[float],
        max_deque_len: int = 100,
        name: str = "SpectrumModel",
        n_chunks: Optional[int] = None,
        map_fn: Callable = map,
        **params,
    ):
        if isinstance(data, str):
            data = Spectrum.load(data)

        self.emulator = emulator

        self.params = FlatterDict()
        self.frozen = []
        self.name = name

        # Unpack the grid parameters
        self.n_grid_params = len(grid_params)
        self.grid_params = grid_params

        self.log = logging.getLogger(self.__class__.__name__)

        if len(data) > 1:
            raise ValueError(
                "Multiple orders detected in data, please use EchelleModel"
            )

        if n_chunks is None:
            n_chunks = mp.cpu_count()

        self.data = data.reshape((n_chunks, -1))
        self.map_fn = map_fn

        self.orders = []
        for i in range(n_chunks):
            self.orders.append(
                OrderModel(
                    emulator,
                    self.data[i],
                    grid_params,
                    max_deque_len,
                    name=f"Order: {i}",
                    **params,
                )
            )

        # Remove order-level params
        params.pop("global_cov", None)
        params.pop("local_cov", None)
        params.pop("cheb", None)
        self.params.update(
            {**params, "orders": [o.independent_params for o in self.orders]}
        )

    @property
    def grid_params(self):
        items = [
            vals
            for key, vals in self.params.items()
            if key in self.emulator.param_names
        ]
        return np.array(items)

    @grid_params.setter
    def grid_params(self, values):
        for i, (key, value) in enumerate(zip(self.emulator.param_names, values)):
            self.params[key] = value

    @property
    def residuals(self):
        return np.asarray([o.residuals for o in self.orders])

    @property
    def labels(self):
        keys = self.get_param_dict(flat=True).keys()
        return list(keys)

    def __call__(self):
        fluxes = []
        covs = []
        for order in self.orders:
            fl, cov = order()
            fluxes.append(fl)
            covs.append(cov)

        return np.asarray(fluxes), np.asarray(covs)

    def log_likelihood(self):
        lnps = self.map_fn(order_likelihood, self.orders)
        return sum(lnps)

    def get_param_dict(self, flat=False):
        params = copy.deepcopy(self.params)
        for key in self.frozen:
            del params[key]

        return params if flat else params.as_dict()

    def set_param_dict(self, params):
        if not isinstance(params, FlatterDict):
            params = FlatterDict(params)

        for key in self.frozen:
            del params[key]

        self.params.update(params)
        order_params = params.pop("orders")
        for order, order_param in zip(self.orders, order_params):
            order.params.update({**params, **order_param})

    def get_param_vector(self):
        values = self.get_param_dict(flat=True).values()
        return np.array(list(values))

    def set_param_vector(self, params):
        params = dict(zip(self.labels, params))
        self.set_param_dict(params)

    def __repr__(self):
        output = f"{self.name}\n"
        output += "-" * len(self.name) + "\n"
        output += f"Data: {self.data.name}\n"
        output += "Parameters:\n"
        for key, value in self.params.as_dict().items():
            output += f"{key}: {value}\n"
        output += f"\nLog Likelihood: {self.log_likelihood()}"
        return output
