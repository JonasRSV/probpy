import unittest
import probpy as pp
import numpy as np
import random
import time
import os


class MyTestCase(unittest.TestCase):
    def test_something_else(self):
        diag_mean = np.array([-1, 1])
        samples = np.concatenate([
            pp.multivariate_normal.sample(mu=np.ones(2) * -3, sigma=np.eye(2), size=2000),
            pp.multivariate_normal.sample(mu=np.ones(2) * 3, sigma=np.eye(2), size=2000),
            pp.multivariate_normal.sample(mu=np.ones(2) * 0, sigma=np.eye(2), size=2000),
            pp.multivariate_normal.sample(mu=diag_mean * -3, sigma=np.eye(2), size=2000),
            pp.multivariate_normal.sample(mu=diag_mean * 3, sigma=np.eye(2), size=2000),
        ])

        timestamp = time.time()
        rv = pp.points.med(points=samples)
        print("Training points", time.time() - timestamp)

        timestamp = time.time()
        probabilities = rv.p(samples)
        print("Probabilities ", time.time() - timestamp)

        modes = pp.algorithms.mode_from_points(
            samples=samples[:100],
            probabilities=probabilities[:100],
            n=100
        )

        timestamp = time.time()
        modes = pp.algorithms.mode_from_points(
            samples=samples,
            probabilities=probabilities,
            n=100
        )
        print("Modes ", time.time() - timestamp)

        print(modes)



if __name__ == '__main__':
    unittest.main()
