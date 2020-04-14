from typing import Callable as F
from typing import Tuple
import numpy as np
from probpy.array_utils import _reshape_samples


def _renormalize_log(log_probs: np.ndarray):
    probability = np.exp(log_probs + np.abs(log_probs.max()))
    probability = np.nan_to_num(probability, copy=False, nan=1e-15,
                                neginf=1e-15, posinf=1e-15)
    return probability / probability.sum()


def hash_samples(samples, volume):
    samples = np.round(_reshape_samples(samples) * volume)

    return [hash(sample.tostring()) for sample in samples]


def search_posterior_estimation(
        size: int,
        log_likelihood: F[[Tuple[np.ndarray]], np.ndarray],
        log_priors: Tuple[F[[np.ndarray], np.ndarray]],
        initial: Tuple[np.ndarray],
        energies: Tuple[float],
        volume: float):
    """

    :param size: number of points to include in search
    :param log_likelihood: log likelihoods
    :param log_priors: logpriors
    :param initial: initial points of search
    :param energies: variance of search jumps
    :param volume: volume of a point in search
    :return: samples and densities
    """
    if len(log_priors) != len(initial): raise Exception(f"log_priors: {len(log_priors)} != initial: {len(initial)}")
    batch = initial[0].shape[0]

    for i in initial:
        if i.shape[0] != batch: raise Exception(f"batch mismatch {batch} != {i.shape[0]}")

    jumpers, n = [], len(initial)
    for i, e in zip(initial, energies): jumpers.append(lambda i=i, e=e: np.random.normal(0, e, size=i.shape))

    def _probability(x: Tuple[np.ndarray]):
        prior_log_probability = np.sum([log_priors[i](x[i]) for i in range(n)], axis=0).flatten()
        data_log_probability = log_likelihood(*x).flatten()

        return prior_log_probability + data_log_probability

    samples = initial
    observations = set()
    indexes = np.arange(0, batch)
    results = [[] for _ in range(n)]
    densities = []
    j = 0
    while len(observations) < size:
        samples = [
            samples[i] + jumpers[i]()
            for i in range(n)
        ]

        density = _probability(samples)

        hashes = hash_samples(samples, volume)

        accept, reject = [], []
        for i, hash in enumerate(hashes):
            if hash in observations:
                reject.append(i)
            else:
                accept.append(i)
                observations.add(hash)

        accept, reject = np.array(accept, dtype=np.int), np.array(reject, dtype=np.int)

        density[reject] = -1e100
        probability = _renormalize_log(density)

        survivors = np.random.choice(indexes, size=batch, p=probability)

        for i in range(n):
            results[i].extend(samples[i][accept])

        densities.extend(density[accept])

        if j % 10 == 0:
            samples = [samples[i][survivors] for i in range(n)]

        j += 1

    densities = np.array(densities)
    densities = _renormalize_log(densities)

    return _reshape_samples(results), densities
