import numpy as np

from probpy.core import Distribution, RandomVariable, Density
from probpy.density import RCKD


class Points(Distribution):
    """Distribution of points"""
    @classmethod
    def med(cls, points: np.ndarray = None,
            variance: float = 2.0,
            error: float = 1e-1,
            verbose: bool = False,
            density_estimator: Density = RCKD) -> RandomVariable:
        """

        :param points: points to estimate density from
        :param variance: variance of kernel
        :param error: acceptable error of partition function
        :param verbose: print estimation of partition function
        :param density_estimator: density estimator
        :return:
        """

        density = density_estimator(variance=variance, error=error, verbose=verbose)
        density.fit(points)

        def _sample(size: int = 1): return points[np.random.randint(low=0, high=points.shape[0], size=size)]
        def _p(x: np.ndarray): return density.p(x)

        parameters = {}
        return RandomVariable(_sample, _p, shape=None, parameters=parameters, cls=cls)

    @staticmethod
    def sample(*arg, **kwargs) -> np.ndarray:
        raise NotImplementedError("Static call of Points is not supported")

    @staticmethod
    def p(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Static call of Points is not supported")
