from abc import abstractmethod, ABC
import numpy as np


class RandomVariable:
    def __init__(self, _sample, _p, shape=None):
        self._sample = _sample
        self._p = _p
        self.shape = shape

    def sample(self, *args, shape: np.ndarray = ()):
        return self._sample(*args, shape=shape)

    def p(self, x, *args):
        return self._p(x, *args)


class Distribution(ABC):

    @classmethod
    @abstractmethod
    def freeze(cls, **kwargs) -> RandomVariable:
        raise NotImplementedError(f"Freeze is not implemented for {cls.__name__}")

    @staticmethod
    @abstractmethod
    def sample(*args, shape: np.ndarray = ()) -> np.ndarray:
        raise NotImplementedError(f"sample is not implemented")

    @staticmethod
    @abstractmethod
    def p(x: np.ndarray, *args) -> np.ndarray:
        raise NotImplementedError(f"pmf / pdf is not implemented")
