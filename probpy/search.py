from typing import Callable as F
import numpy as np


def _renormalize_log(log_probs: np.ndarray):
    probability = np.exp(log_probs + np.abs(log_probs.max()))
    probability = np.nan_to_num(probability, copy=False, nan=1e-15,
                                neginf=1e-15, posinf=1e-15)
    return probability / probability.sum()


def hash_samples(samples):
    return [hash(sample.tostring()) for sample in np.round(samples)]


def search_posterior_estimation(
        size: int,
        log_likelihood: F[[np.ndarray], np.ndarray],
        log_prior: F[[np.ndarray], np.ndarray],
        initial: np.ndarray,
        energy: float,
        volume: float):
    """

    :param size: number of points to include in search
    :param log_likelihood: log likelihoods
    :param log_prior: log prior
    :param initial: initial points of search
    :param energy: variance of search jumps
    :param volume: volume of a point in search
    :return: samples and densities
    """
    batch = initial.shape[0]
    jump = lambda: np.random.normal(0, energy, size=initial.shape)

    def _probability(x: np.ndarray):
        log_prior_probability = np.nan_to_num(log_prior(x).flatten(), nan=-15000, posinf=-15000, neginf=-15000)
        log_likelihood_probability = np.nan_to_num(log_likelihood(x).flatten(), nan=-15000, posinf=-15000, neginf=-15000)

        return log_prior_probability + log_likelihood_probability

    p = initial
    results, densities, observations, indexes = [], [], set(), np.arange(0, batch)
    while len(results) < size:
        samples = p + jump()

        density = _probability(samples)
        hashes = hash_samples(samples * volume)

        accept, reject = [], []
        for i, hash in enumerate(hashes):
            if hash in observations:
                reject.append(i)
            else:
                accept.append(i)
                observations.add(hash)

        accept, reject = np.array(accept, dtype=np.int), np.array(reject, dtype=np.int)

        density[reject] = -1e100

        accept_rate = np.minimum(density - _probability(p), 0.0)
        survivors = accept_rate >= np.log(np.random.rand(batch))

        results.extend(samples[accept])
        densities.extend(density[accept])

        p[survivors] = samples[survivors]

    densities = np.array(densities)
    densities = _renormalize_log(densities)

    return np.array(results), densities
