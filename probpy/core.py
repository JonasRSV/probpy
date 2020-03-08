from abc import abstractmethod, ABC
import numpy as np


class Distribution(ABC):
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
