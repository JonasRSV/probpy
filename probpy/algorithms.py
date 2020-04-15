import numpy as np
import heapq
import numba


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def euclidean(i, j):
    return np.sqrt(np.square(i - j).sum())


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _closest_n(x: np.ndarray, probabilities: np.ndarray, samples: np.ndarray, n: int):
    heap = [(np.float(i), np.int(i)) for i in range(0)]
    for i in range(probabilities.size):
        heapq.heappush(heap, (-euclidean(samples[i], samples[x]), i))

        if len(heap) > n:
            heapq.heappop(heap)

    return [j[1] for j in heap]


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _mode_climb(current: np.int,
                cache: np.ndarray,
                probabilities: np.ndarray,
                samples: np.ndarray,
                n: int):
    if cache[current]:
        return -1, -1.0

    cache[current] = True

    while True:
        closest = _closest_n(current, probabilities, samples, n)

        max_i, max_p = current, probabilities[current]
        for i in closest:
            if probabilities[i] > max_p:
                max_i, max_p = i, probabilities[i]

            cache[i] = True

        if max_i == current:
            break

        current = max_i

    return current, probabilities[current]


@numba.jit(nopython=True, forceobj=False, fastmath=True)
def _hacky_numpy_hash(x: np.ndarray) -> float: # TODO make this deal with high dimensional arrays better
    _hash: float = 0.0
    for i in range(x.size):
        _hash += x[i] * float(i % 10)
    return _hash


@numba.jit(nopython=True, forceobj=False)
def _sort_modes(modes: np.ndarray, densities: np.ndarray):
    sz = len(densities)

    densities = [i for i in densities]
    indexes = np.arange(0, sz)

    for i in range(sz):
        for j in range(sz):
            if densities[i] > densities[j]:
                indexes[i], indexes[j] = indexes[j], indexes[i]
                densities[i], densities[j] = densities[j], densities[i]

    return [modes[i] for i in indexes], densities


@numba.jit(nopython=True, forceobj=False)
def mode_from_points(samples: np.ndarray, probabilities: np.ndarray, n=100):
    visited = numba.typed.List()
    [visited.append(False) for _ in range(samples.shape[0])]

    observed = set([0.0 for _ in range(0)])
    modes = []
    densities = []
    for i in range(probabilities.size):
        _mode, p = _mode_climb(i, visited, probabilities, samples, n)

        if p != -1.0:
            _mode_value = samples[_mode].flatten()
            _mode_hash: float = _hacky_numpy_hash(_mode_value)

            if p > 0.0:
                if _mode_hash in observed: # "not in" does not work in numba
                    pass
                else:
                    modes.append(_mode_value)
                    densities.append(p)

            observed.add(_mode_hash)

    _modes, _densities = _sort_modes(modes, densities)

    return _modes, _densities
