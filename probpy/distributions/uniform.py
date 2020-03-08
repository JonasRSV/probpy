import numpy as np

from probpy.core import Distribution


class Uniform(Distribution):

    @staticmethod
    def sample(a: np.ndarray, b: np.ndarray, shape: np.ndarray = ()) -> np.ndarray:
        if type(shape) == int:
            return a + np.random.rand(shape) * (b - a)

        return a + np.random.rand(*shape) * (b - a)

    @staticmethod
    def p(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        return ((a < x) & (x < b)).astype(np.float32) / (b - a)


class MultiVariateUniform(Distribution):
    @staticmethod
    def sample(a: np.ndarray, b: np.ndarray, shape: np.ndarray = ()) -> np.ndarray:
        if type(shape) == int:
            return a + np.random.rand(shape, a.shape[0]) * (b - a)

        if shape != ():
            if a.shape != shape[1:]:
                raise Exception(f"shape of a needs to match provided shape a: {a.shape} -- provided: {shape} -- {a.shape} != {shape[1:]}")

            return a + np.random.rand(*shape) * (b - a)

        return a + np.random.rand(a.shape) * (b - a)

    @staticmethod
    def p(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        return ((a < x) & (x < b)).all(axis=1).astype(np.float32) / np.product(b - a)


