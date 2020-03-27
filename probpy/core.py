from typing import Tuple, Dict, Union
from abc import abstractmethod, ABC
import numpy as np


class Parameter:
    def __init__(self, shape: Union[Tuple, int], value: np.ndarray):
        self.shape = shape
        self.value = value


class RandomVariable:
    def __init__(self, _sample, _p,
                 shape=None,
                 parameters: Dict[str, Parameter] = {},
                 cls: object = None):
        self._sample = _sample
        self._p = _p
        self.shape = shape
        self.parameters = parameters
        self.cls = cls

        for name, parameter in parameters.items():
            self.__setattr__(name, parameter.value)

    def sample(self, *args, shape: np.ndarray = ()):
        return self._sample(*args, shape=shape)

    def p(self, x, *args):
        return self._p(x, *args)

    def __str__(self):
        title = f"{self.cls.__name__} -- output: {self.shape}\n"
        body = '\n'.join(
            [f'{name}: {parameter.shape} - {parameter.value}' for name, parameter in self.parameters.items()])
        return title + body


class Distribution(ABC):

    @classmethod
    @abstractmethod
    def med(cls, **kwargs) -> RandomVariable:
        raise NotImplementedError(f"Freeze is not implemented for {cls.__name__}")

    @staticmethod
    @abstractmethod
    def sample(*args, shape: np.ndarray = ()) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError(f"sample is not implemented")

    @staticmethod
    @abstractmethod
    def p(x: np.ndarray, *args) -> np.ndarray:
        raise NotImplementedError(f"pmf / pdf is not implemented")


class Density(ABC):

    @abstractmethod
    def fit(self, particles: np.ndarray):
        raise NotImplementedError()

    def p(self, particles: np.ndarray):
        raise NotImplementedError()
