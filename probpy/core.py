from abc import abstractmethod, ABC
import numpy as np


class FrozenDistribution():
    def __init__(self, distribution: "Distribution", *args):
        self.distribution = distribution
        self.args = args

    def sample(self, shape: np.ndarray = ()):
        return self.distribution.sample(*self.args, shape=shape)

    def p(self, x: np.ndarray):
        return self.distribution.p(*self.args, x)


class Distribution(ABC):

    @classmethod
    @abstractmethod
    def freeze(cls, *args) -> FrozenDistribution:
        return FrozenDistribution(cls, *args)

    @staticmethod
    @abstractmethod
    def sample(*args, shape: np.ndarray = ()) -> np.ndarray:
        raise NotImplementedError(f"sample is not implemented")

    @staticmethod
    @abstractmethod
    def p(*args, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"pmf / pdf is not implemented")


class BBN(ABC):
    pass
    # network.add_edge(
