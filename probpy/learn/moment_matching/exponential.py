import numpy as np
from probpy.distributions import exponential


def exponential_matcher(samples: np.ndarray):
    mean = samples.mean()
    return exponential.med(lam=1 / mean)


