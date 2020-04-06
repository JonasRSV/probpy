import unittest
import numpy as np
import probpy as pp


class VisualTestMode(unittest.TestCase):
    def test_mode_points(self):
        diag_mean = np.array([-1, 1])
        samples = np.concatenate([
            pp.multivariate_normal.sample(mu=np.ones(2) * -3, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=np.ones(2) * 3, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=np.ones(2) * 0, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=diag_mean * -3, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=diag_mean * 3, sigma=np.eye(2), size=1000),
        ])

        distribution = pp.distributions.points.med(points=samples)

        print(len(pp.mode(distribution)))
        print(pp.mode(distribution))


if __name__ == '__main__':
    unittest.main()
