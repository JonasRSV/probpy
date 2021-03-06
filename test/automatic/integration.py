import unittest

from probpy.integration import uniform_importance_sampling, expected_value, posterior_predictive_integration
from probpy.distributions import *
import numpy as np


class TestIntegration(unittest.TestCase):

    def test_uniform_importance_sampling(self):
        f = lambda x: -np.square(x) + 3

        result = uniform_importance_sampling(size=10000000,
                                             function=f,
                                             domain=(-2, 2),
                                             proposal=normal.med(mu=0, sigma=2))

        self.assertAlmostEqual(result, 6.666, delta=0.1)

        f = lambda x: -np.square(x[:, 0]) + np.square(x[:, 1])

        lower_bound = np.array([0, 0])
        upper_bound = np.array([4, 2])

        result = uniform_importance_sampling(size=100000,
                                             function=f,
                                             domain=(lower_bound, upper_bound),
                                             proposal=multivariate_normal.med(mu=np.zeros(2),
                                                                              sigma=np.eye(2) * 2))
        self.assertAlmostEqual(result, -32, delta=3.0)

    def test_posterior_predictive_integral(self):
        def _run_test(priors=None, likelihood=None, correct=None):
            result = posterior_predictive_integration(500, likelihood, priors)

            if correct is not None:
                pass # TODO

        tests = [
            {
                "priors": (normal.med(mu=0.0, sigma=1.0), exponential.med(lam=1.0)),
                "likelihood": lambda *theta: normal.med().p(0.5, *theta),
                "correct": None
            }
        ]

        for test in tests:
            _run_test(**test)

    def test_expected_value(self):

        function = lambda x: x
        tests = [
            (normal.med(mu=0, sigma=1), 0.0),
            (multivariate_normal.med(mu=np.zeros(2), sigma=np.eye(2)), np.zeros(2)),
            (exponential.med(lam=1.0), 1.0),
            (bernoulli.med(probability=0.5), 0.5),
            (beta.med(a=2.0, b=2.0), 0.5),
            (dirichlet.med(alpha=np.ones(5)), np.ones(5) / 5),
            (poisson.med(lam=1.0), 1.0),
            (uniform.med(a=0, b=1.0), 0.5)
        ]

        size = int(1e5)
        for dist, label in tests:
            expectation = expected_value(size, function, dist)

            if expectation.ndim == 0:
                self.assertAlmostEqual(expectation, label, delta=0.1)
            elif expectation.ndim == 1:
                for i in range(expectation.size):
                   self.assertAlmostEqual(expectation[i], label[i], delta=0.1)
            else:
                raise Exception(f"Samples should not have {expectation.dim} dim")


if __name__ == '__main__':
    unittest.main()
