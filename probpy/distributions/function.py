import numpy as np

from probpy.core import Distribution, RandomVariable
from probpy.density import RCKD
from probpy.sampling import fast_metropolis_hastings
from probpy.distributions import multivariate_uniform


class Function(Distribution):
    @classmethod
    def med(cls, density,
            lower_bound: np.ndarray,
            upper_bound: np.ndarray,
            points: int = 1000,
            variance: float = 2.0,
            error: float = 1e-1,
            batch: int = 25,
            verbose: bool = False) -> RandomVariable:

        lower_bound, upper_bound = np.array(lower_bound), np.array(upper_bound)
        initial = multivariate_uniform.sample(lower_bound, upper_bound, size=batch)

        samples = fast_metropolis_hastings(np.maximum(points, 10000), density, initial=initial)[-points:]

        density = RCKD(variance=variance, sampling_sz=100, error=error, verbose=verbose)
        density.fit(samples)

        def _sample(size: int = 1): return samples[np.random.randint(low=0, high=samples.shape[0], size=size)]
        def _p(x: np.ndarray): return density.p(x)

        parameters = {}
        return RandomVariable(_sample, _p, shape=None, parameters=parameters, cls=cls)

    @staticmethod
    def sample(*arg, **kwargs) -> np.ndarray:
        raise NotImplementedError("Static call of Function is not supported")

    @staticmethod
    def p(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Static call of Function is not supported")
