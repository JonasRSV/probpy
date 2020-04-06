from typing import Callable as F, List
from typing import Tuple
import numpy as np


def _reshape_samples(samples: List[np.ndarray]):  # cannot import from learn because of ciruclar deps
    result = []
    for sample in samples:
        if sample.ndim == 1:
            result.append(sample.reshape(-1, 1))
        else:
            result.append(sample)
    return result


def fast_almost_mcmc_parameter_posterior_estimation(
        size: int,
        log_likelihood: F[[Tuple[np.ndarray]], np.ndarray],
        log_priors: Tuple[F[[np.ndarray], np.ndarray]],
        initial: Tuple[np.ndarray],
        energies: Tuple[float]):
    if len(log_priors) != len(initial): raise Exception(f"log_priors: {len(log_priors)} != initial: {len(initial)}")
    batch = initial[0].shape[0]

    for i in initial:
        if i.shape[0] != batch: raise Exception(f"batch mismatch {batch} != {i.shape[0]}")

    jumpers, n = [], len(initial)
    for i, e in zip(initial, energies): jumpers.append(lambda i=i, e=e: np.random.normal(0, e, size=i.shape))

    def _probability(x: Tuple[np.ndarray]):
        prior_log_probability = np.sum([log_priors[i](x[i]) for i in range(n)], axis=0)
        data_log_probability = log_likelihood(*x)
        return prior_log_probability + data_log_probability

    p = initial
    points = [[] for _ in range(n)]
    densities = []
    while len(points[0]) < size:
        samples = [
            p[i] + jumpers[i]()
            for i in range(n)
        ]

        density = _probability(samples)

        accept_rate = np.minimum(density - _probability(p), 0.0)
        accept_rate = accept_rate.flatten()

        accepted = accept_rate >= np.log(np.random.rand(batch))
        rejected = False == accepted

        for i in range(n):
            points[i].extend(samples[i][accepted])
            samples[i][rejected] = p[i][rejected]

        densities.extend(density[accepted])

        p = samples

    densities = np.array(densities)
    return np.concatenate(_reshape_samples([np.array(point) for point in points]), axis=1), \
           np.exp(densities + np.abs(densities.max()))
