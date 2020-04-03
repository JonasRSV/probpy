import unittest
from probpy.distributions import (normal,
                                  multivariate_normal,
                                  uniform,
                                  multivariate_uniform,
                                  exponential,
                                  beta,
                                  bernoulli,
                                  categorical,
                                  dirichlet,
                                  gamma,
                                  normal_inverse_gamma,
                                  binomial,
                                  multinomial,
                                  poisson,
                                  geometric,
                                  gaussian_process,
                                  unilinear)
from probpy.inference import predictive_posterior
import numpy as np


class PredictivePosteriorTest(unittest.TestCase):
    def test_conjugate_analytic(self):
        def _run_test(prior=None, likelihood=None, correct=None):
            result = predictive_posterior(likelihood=likelihood, priors=prior)

            if correct is not None:
                pass  # TODO

        tests = ({
                     "prior": beta.med(a=8.0, b=3.0),
                     "likelihood": bernoulli.med(),
                     "correct": None
                 },
                 {
                     "prior": normal.med(mu=1.0, sigma=1.0),
                     "likelihood": normal.med(sigma=1.0),
                     "correct": None
                 },
                 {
                     "prior": multivariate_normal.med(np.zeros(2), sigma=np.eye(2)),
                     "likelihood": multivariate_normal.med(sigma=np.eye(2)),
                     "correct": None
                 })

        for test in tests:
            _run_test(**test)

    def test_numerical_integration_probabilities(self):
        def _run_test(prior=None, likelihood=None, data=None, correct=None):
            result = predictive_posterior(likelihood=likelihood, priors=prior, data=data)

            if correct is not None:
                pass  # TODO

        def get_grid_data(lb, ub, size):
            grid = size
            i = np.linspace(lb, ub, grid)
            j = np.linspace(lb, ub, grid)

            I, J = np.meshgrid(i, j)
            X = np.concatenate([I.reshape(-1, 1), J.reshape(-1, 1)], axis=1)

            return X

        def _custom_likelihood(y, x, w):
            result = []
            for _w in w:
                result.append(
                    normal.p(_w - np.float_power(y, x), mu=0.0, sigma=0.01)
                )

            return np.array(result)

        tests = ({
                     "prior": uniform.med(a=0.0, b=3.0),
                     "likelihood": normal.med(sigma=0.5),
                     "data": np.linspace(-2, 4, 20),
                     "correct": None

                 }, {
                     "prior": multivariate_uniform.med(a=np.ones(2) * 0, b=np.ones(2) * 2),
                     "likelihood": multivariate_normal.med(sigma=np.eye(2)),
                     "data": get_grid_data(-2, 2, 20),
                     "correct": None
                 }, {
                     "prior": multivariate_normal.med(mu=np.ones(2), sigma=np.eye(2)),
                     "likelihood": unilinear.med(sigma=0.3),
                     "data": (np.linspace(-2, 4, 20), np.array(0.8)),
                     "correct": None
                 }, {
                     "prior": uniform.med(a=-4, b=4),
                     "likelihood": _custom_likelihood,
                     "data": (np.linspace(-3, 3, 100), np.array(2.0)),
                     "correct": None
                 })

        for test in tests:
            _run_test(**test)


if __name__ == '__main__':
    unittest.main()
