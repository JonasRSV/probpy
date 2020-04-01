import numpy as np
import numba

from probpy.core import Distribution, RandomVariable, Parameter
from probpy.distributions.binomial import Binomial


class Hypergeometric(Distribution):
    N = "N"
    K = "K"
    n = "n"

    @classmethod
    def med(cls, N: np.int = None, K: np.int = None, n: np.int = None) -> RandomVariable:
        params = [N, K, n]
        none = [i for i, param in enumerate(params) if param is None]
        not_none = [i for i, param in enumerate(params) if param is not None]

        def _p(x, *args):
            call_args = [None] * 3
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return Hypergeometric.p(x, *call_args)

        def _sample(*args, size=()):
            call_args = [None] * 3
            for i, arg in enumerate(args): call_args[none[i]] = arg
            for i in not_none: call_args[i] = params[i]

            return Hypergeometric.sample(*call_args, size=size)

        parameters = {
            Hypergeometric.N: Parameter((), N),
            Hypergeometric.K: Parameter((), K),
            Hypergeometric.n: Parameter((), n)
        }

        return RandomVariable(_sample, _p, shape=(), parameters=parameters, cls=cls)

    @staticmethod
    @numba.jit(nopython=False, forceobj=True)
    def sample(N: np.int, K: np.int, n: np.int, size = ()) -> np.ndarray:
        return np.random.hypergeometric(ngood=K, nbad=N - K, nsample=n, size=size)

    @staticmethod
    def p(x: np.ndarray, N: np.int, K: np.int, n: np.int) -> np.ndarray:
        K_choose_x = np.array([Binomial._combinations_high_n(K, _x) for _x in x])
        NK_choose_nx = np.array([Binomial._combinations_high_n(N - K, n - _x) for _x in x])
        N_choose_n = np.array(Binomial._combinations_high_n(N, n))
        return (K_choose_x * NK_choose_nx) / N_choose_n
