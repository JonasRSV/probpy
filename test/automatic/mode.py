import unittest
import numpy as np
import probpy as pp


class TestModes(unittest.TestCase):
    def test_modes(self):
        diag_mean = np.array([-1, 1])
        samples = np.concatenate([
            pp.multivariate_normal.sample(mu=np.ones(2) * -3, sigma=np.eye(2), size=200),
            pp.multivariate_normal.sample(mu=np.ones(2) * 3, sigma=np.eye(2), size=200),
            pp.multivariate_normal.sample(mu=np.ones(2) * 0, sigma=np.eye(2), size=200),
            pp.multivariate_normal.sample(mu=diag_mean * -3, sigma=np.eye(2), size=200),
            pp.multivariate_normal.sample(mu=diag_mean * 3, sigma=np.eye(2), size=200),
        ])

        tests = [
            {
                "distribution": pp.normal.med(mu=0.0, sigma=1.0),
                "correct": lambda x: x == 0.0
            },
            {
                "distribution": pp.multivariate_normal.med(np.zeros(2), np.eye(2)),
                "correct": lambda x: all(x == np.zeros(2))
            },
            {
                "distribution": pp.gamma.med(a=1.0, b=2.0),
                "correct": lambda _: True,
            },
            {
                "distribution": pp.beta.med(a=1.0, b=2.0),
                "correct": lambda _: True
            },
            {
                "distribution": pp.points.med(points=samples),
                "correct": lambda x: len(x) == 5
            },
            {
                "distribution": pp.dirichlet.med(alpha=np.ones(5)),
                "correct": lambda _: True
            },
            {
                "distribution": pp.categorical.med(probabilities=np.ones(5) / 5),
                "correct": lambda _: True
            },
            {
                "distribution": pp.normal_inverse_gamma.med(mu=1.0, lam=2.0, a=2.0, b=2.0),
                "correct": lambda _: True
            }
        ]

        for test in tests:
            self.assertTrue(test["correct"](pp.mode(test["distribution"], n=10)))


if __name__ == '__main__':
    unittest.main()
