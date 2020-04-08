import numpy as np
import heapq


def euclidean(i, j):
    return np.sqrt(np.square(i - j).sum())


def mode_from_points(samples, probabilities, n=100, distance=euclidean):
    visited = [False] * samples.shape[0]

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

            visited[i] = True

        if x == maxi:
            return maxi, maxp

        visited[maxi] = False
        return _mode_climb(maxi)

    observed = set()
    modes = []

    for i, sample in enumerate(samples):
        _mode, p = _mode_climb(i)

        if _mode is not None and p > 0 and samples[_mode].tostring() not in observed:
            modes.append((samples[_mode].reshape(-1), p))

        observed.add(samples[_mode].tostring())

    modes = sorted(modes, key=lambda x: x[1], reverse=True)

    modes, densities = zip(*modes)

    return np.array(modes), np.array(densities)
