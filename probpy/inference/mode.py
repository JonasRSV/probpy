from probpy.core import RandomVariable
from probpy.distributions import *
import numpy as np
import heapq


def normal_mode(rv: RandomVariable, **_): return rv.mu


def multivariate_normal_mode(rv: RandomVariable, **_): return rv.mu


def gamma_mode(rv: RandomVariable, **_): return (rv.a - 1) / rv.b


def beta_mode(rv: RandomVariable, **_): return (rv.a - 1) / (rv.a + rv.b - 2)


def dirichlet_mode(rv: RandomVariable, **_): return (rv.alpha - 1) / (rv.alpha.sum() - rv.alpha.size)


def categorical_mode(rv: RandomVariable, **_): return np.eye(rv.probabilities.size)[np.argmax(rv.probabilities)]


def normal_inverse_gamma_mode(rv: RandomVariable, **_): return np.array([rv.mu, rv.b / (rv.a + 3 / 2)])


def euclidean(i, j):
    return np.sqrt(np.square(i - j).sum())


def points_mode(rv: RandomVariable, samples=100, n=10, distance=euclidean):
    visited = [False] * samples
    samples = rv.sample(size=samples)

    probabilities = rv.p(samples)

    def closest_n(x: np.ndarray):
        heap = []
        for i, sample in enumerate(samples):
            heapq.heappush(heap, (-distance(sample, samples[x]), i))

            if len(heap) > n:
                heapq.heappop(heap)

        return list(map(lambda j: j[1], heap))

    def _mode_climb(x: np.ndarray):
        if visited[x]:
            return None, None

        visited[x] = True

        closest = closest_n(x)

        maxi, maxp = x, probabilities[x]

        for i in closest:
            if probabilities[i] > maxp:
                maxi, maxp = i, probabilities[i]

        if x == maxi:
            return maxi, maxp

        return _mode_climb(maxi)

    modes = []
    for i, sample in enumerate(samples):
        _mode, p = _mode_climb(i)

        if _mode is not None:
            modes.append((samples[_mode], p))

    modes = sorted(modes, key=lambda x: x[1], reverse=True)

    return list(map(lambda x: x[0], modes))


implemented = {
    normal: normal_mode,
    multivariate_normal: multivariate_normal_mode,
    gamma: gamma_mode,
    beta: beta_mode,
    points: points_mode,
    dirichlet: dirichlet_mode,
    categorical: categorical_mode,
    normal_inverse_gamma: normal_inverse_gamma_mode
}


def mode(rv: RandomVariable, **kwargs):
    if rv.cls in implemented:
        return implemented[rv.cls](rv, **kwargs)

    raise NotImplementedError(f"Mode not implemented for {rv.cls.__class__}")
