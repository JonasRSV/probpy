from typing import Callable as F
from typing import List
import numpy as np
from .core import RandomVariable


def metropolis_hastings(size: int,
                        pdf: F[[np.ndarray], np.ndarray],
                        proposal: RandomVariable,
                        initial: np.ndarray = None) -> List[np.ndarray]:
    if initial is None:
        p = np.random.rand(*proposal.shape)
    else:
        p = initial

    samples = []
    while len(samples) < size:
        sample = proposal.sample(p)

        accept_rate = min([(pdf(sample) * proposal.p(p, sample)) / (pdf(p) * proposal.p(sample, p)), 1])
        if np.random.rand() < accept_rate:
            samples.append(sample)
            p = sample

    return samples


def metropolis(size: int,
               pdf: F[[np.ndarray], np.ndarray],
               proposal: RandomVariable,
               M: float) -> List[np.ndarray]:
    samples = []
    while len(samples) < size:
        remainder = size - len(samples)
        sample = proposal.sample(shape=remainder)

        accept_rate = pdf(sample) / (M * proposal.p(sample))

        max_rate = accept_rate.max()
        if max_rate > 1.0: raise Exception(f"M to small, accept rate {max_rate} > 1.0. m: {M} ")

        rejection_probability = np.random.rand(remainder)
        accept_mask = accept_rate > rejection_probability

        samples.extend(sample[accept_mask])

    return samples
