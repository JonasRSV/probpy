import numpy as np
from typing import List


def _reshape_samples(samples: List[np.ndarray]):
    result = []
    for sample in samples:
        sample = np.array(sample)
        if sample.ndim == 1:
            result.append(sample.reshape(-1, 1))
        else:
            result.append(sample)

    return np.concatenate(result, axis=1)
