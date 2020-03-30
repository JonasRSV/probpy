import numpy as np

from probpy.core import Distribution, RandomVariable
from probpy.density import RCKD
from probpy.mcmc import fast_metropolis_hastings


class Function(Distribution):
    @classmethod
    def med(cls, density,
            initial: np.ndarray,
            points: int = 1000,
            variance: float = 2.0,
            error: float = 1e-1,
            verbose: bool = False) -> RandomVariable:

        samples = fast_metropolis_hastings(points, density, initial)

        density = RCKD(variance=variance, sampling_sz=100, error=error, verbose=verbose)
        density.fit(samples)

        def _sample(shape: np.ndarray = ()): return samples[np.random.randint(low=0, high=samples.shape[0], size=shape)]
        def _p(x: np.ndarray): return density.p(x)

        parameters = {}
        return RandomVariable(_sample, _p, shape=None, parameters=parameters, cls=cls)

    @staticmethod
    def sample(*arg, **kwargs) -> np.ndarray:
        raise NotImplementedError("Static call of Function is not supported")

    @staticmethod
    def p(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Static call of Function is not supported")
