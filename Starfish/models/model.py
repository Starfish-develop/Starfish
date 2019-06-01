from typing import Union, Sequence, List, Tuple
from collections import OrderedDict

from nptyping import Array
import numpy as np

from Starfish.emulator import Emulator
from Starfish.spectrum import Spectrum


class Model:
    def __init__(
        self,
        emulator: Union[str, Emulator],
        spectrum: Union[str, Spectrum],
        grid_params: Sequence[float],
        max_deque_len: int = 100,
        name: str = "Model",
    ):
        if isinstance(emulator, str):
            emulator = Emulator.load(emulator)
        self.emulator = emulator

        if isinstance(spectrum, str):
            spectrum = Spectrum.load(spectrum)
        self.spectrum = spectrum
        self.params = OrderedDict()
        self.frozen = []
        self.name = name

    def __call__(self) -> Tuple[Array[float], Array[float]]:
        raise NotImplementedError("Needs to be implemented by subclasses")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    @property
    def labels(self) -> List[str]:
        return list(self.get_param_dict())

    def get_param_dict(self) -> dict:
        params = {}
        for key, val in self.params.items():
            if key not in self.frozen:
                params[key] = val

        return params

    def set_param_dict(self, params: dict):
        for key, val in params.items():
            if key not in self.frozen:
                self.params[key] = val

    def get_param_vector(self) -> Array[float]:
        return self.get_param_dict().values()

    def set_param_vector(self, params: Array[float]):
        self.set_param_dict(zip(self.labels, params))

    def freeze(self, names: Union[str, Sequence[str]]):
        """
        Freeze the given parameter such that :meth:`get_param_dict` and :meth:`get_param_vector` no longer include this parameter, however it will still be used when calling the model.

        Parameters
        ----------
        name : str or array-like
            The parameter to freeze. If 'all', will freeze all parameters.

        Raises
        ------
        ValueError
            If the given parameter does not exist

        See Also
        --------
        :meth:`thaw`
        """
        names = np.atleast_1d(names)
        if names[0] == "all":
            self.frozen.append(self.labels)
        else:
            for name in names:
                if name not in self.frozen:
                    self.frozen.append(name)

    def thaw(self, names: Union[str, Sequence[str]]):
        """
        Thaws the given parameter. Opposite of freezing

        Parameters
        ----------
        name : str or array-like
            The parameter to thaw. If 'all', will thaw all parameters

        Raises
        ------
        ValueError
            If the given parameter does not exist.

        See Also
        --------
        :meth:`freeze`
        """
        names = np.atleast_1d(names)
        if names[0] == "all":
            self.frozen = []
        else:
            for name in names:
                if name in self.frozen:
                    self.frozen.remove(name)
