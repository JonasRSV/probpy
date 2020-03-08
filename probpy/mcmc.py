from typing import Callable as F
from typing import List
import numpy as np


def metropolis_hastings(pdf: F[[np.ndarray], np.ndarray],
                        proposal: F[[np.ndarray, np.ndarray], np.ndarray],
                        sampler: F[[np.ndarray], np.ndarray],
                        shape: np.ndarray,
                        p0: np.ndarray = None) -> List[np.ndarray]:
    size, dim = shape[0], shape[1:]
    if p0 is None:
        p0 = np.random.rand(*dim)

    samples = []
    while len(samples) < size:
        sample = sampler(p0)

        accept_rate = min([(pdf(sample) * proposal(sample, p0)) / (pdf(p0) * proposal(p0, sample)), 1])
        if np.random.rand() < accept_rate:
            samples.append(sample)
            p0 = sample

    return samples


def metropolis(pdf: F[[np.ndarray], np.ndarray],
               proposal: F[[np.ndarray], np.ndarray],
               sampler: F[[None], np.ndarray],
               M: float,
               shape: np.ndarray) -> List[np.ndarray]:
    size, dim = shape[0], shape[1:]

    samples = []
    while len(samples) < size:
        sample = sampler()

        accept_rate = pdf(sample) / (M * proposal(sample))
        if accept_rate > 1.0: raise Exception(f"M to small, accept rate {accept_rate} > 1.0. m: {M} ")
        if np.random.rand() < accept_rate:
            samples.append(sample)

    return samples
