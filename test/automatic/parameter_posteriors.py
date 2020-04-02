import unittest
from probpy.core import RandomVariable
from probpy.distributions import (normal,
                                  multivariate_normal,
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
                                  unilinear,
                                  multivariate_uniform)
from probpy.learn import parameter_posterior
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


class PosteriorTest(unittest.TestCase):
    def test_conjugates(self):

        mu_prior, sigma_prior = np.ones(2), np.eye(2)
        probabilities = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        x = np.linspace(0, 1, 10)
        y = x + 1 + normal.sample(mu=0.0, sigma=1e-1, size=10)
        tests = [
            (normal.med(mu=1.0, sigma=1.0), normal.med(sigma=2.0), normal.med(mu=-2.0, sigma=2.0), None),
            (normal_inverse_gamma.med(mu=1.0, lam=2.0, a=3.0, b=3.0), normal.med(), normal.med(mu=-2.0, sigma=2.0),
             None),
            (multivariate_normal.med(mu=mu_prior, sigma=sigma_prior), multivariate_normal.med(sigma=sigma_prior * 10),
             multivariate_normal.med(mu=mu_prior, sigma=sigma_prior), None),
            (beta.med(a=1.0, b=3.0), bernoulli.med(), bernoulli.med(probability=0.6), None),
            (dirichlet.med(alpha=np.ones(5)), categorical.med(dim=5), categorical.med(probabilities=probabilities),
             None),
            (gamma.med(a=9, b=2), exponential.med(), exponential.med(lam=1), None),
            (beta.med(a=6.0, b=3.0), binomial.med(n=5), binomial.med(n=5, probability=0.7), None),
            (dirichlet.med(alpha=np.ones(3)), multinomial.med(n=3), multinomial.med(n=3, probabilities=np.ones(3) / 3),
             None),
            (gamma.med(a=9, b=2), poisson.med(), poisson.med(lam=2), None),
            (beta.med(a=6.0, b=3.0), geometric.med(), geometric.med(probability=0.7), None),
            (multivariate_normal.med(mu=mu_prior, sigma=sigma_prior), unilinear.med(sigma=1e-1), (y, x), None)

        ]

        for priors, likelihood, data, result in tests:
            if type(data) == RandomVariable: data = data.sample(size=100)

            posterior = parameter_posterior(data, likelihood=likelihood, priors=priors)

            if result is not None:
                pass  # TODO

    def test_mcmc(self):

        def _run_test(priors, likelihood, data, correct):
            posterior = parameter_posterior(data, likelihood=likelihood, priors=priors, size=1000)

            if correct is not None:
                pass  # TODO

        mu_prior = np.zeros(2)
        sigma_prior = np.eye(2)
        variables = np.array([2, 1])
        x = np.linspace(-2, 2, 60)
        y = unilinear.sample(x=x, variables=variables, sigma=0.3)
        a = np.zeros(2)
        b = np.ones(2)

        logistic_x = np.linspace(-5, 5, 50).reshape(-1, 1)
        logistic_y = (logistic_x > 0).astype(np.float).flatten()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def _custom_likelihood(y, x, w):
            return normal.p((y - sigmoid(x @ w[:, None, :-1] + w[:, None, None, -1]).squeeze(axis=2)),
                            mu=0.0, sigma=0.5)

        tests = [
            ((exponential.med(lam=1.0),), normal.med(sigma=2.0),
             normal.sample(mu=3.0, sigma=2.0, size=100), None),
            ((normal.med(mu=3.0, sigma=2.0), exponential.med(lam=1.0)), normal.med(),
             normal.sample(mu=5.0, sigma=2.0, size=300), None),
            ((multivariate_uniform.med(a=a, b=b),),
             multivariate_normal.med(sigma=sigma_prior),
             multivariate_normal.sample(mu=mu_prior, sigma=sigma_prior, size=100),
             None),
            ((multivariate_normal.med(mu_prior, sigma=sigma_prior), exponential.med(lam=1.0)),
             unilinear.med(), (y, x), None),
            ((multivariate_normal.med(mu=mu_prior, sigma=sigma_prior),), _custom_likelihood, (logistic_y, logistic_x),
             None)
        ]

        for test in tests:
            _run_test(*test)

    def test_mcmc_moment_matching(self):
        def _run_test(priors, likelihood, data, match, correct):
            posterior = parameter_posterior(data, likelihood=likelihood,
                                            priors=priors, size=1000, match_moments_for=match)

            if correct is not None:
                pass  # TODO

        def _custom_likelihood(y, x, w):
            result = []
            for _w in w: result.append(normal.p(y - x * _w[0] - _w[1], mu=0.0, sigma=0.3))
            return np.array(result)


        a, b = np.zeros(2), np.ones(2) * 4
        sigma = np.eye(2)
        mu = np.zeros(2)
        x = np.linspace(-1, 1, 100)
        y = x * 2 + 0.5 + normal.sample(mu=0.0, sigma=0.3, size=100)

        tests = [
            ((exponential.med(lam=0.6, )), normal.med(sigma=1.0), normal.sample(mu=3.0, sigma=2.0, size=200),
             normal, None),
            ((multivariate_uniform.med(a=a, b=b), ), multivariate_normal.med(sigma=sigma),
             multivariate_normal.sample(mu=mu, sigma=sigma, size=200),
             multivariate_normal, None),
            ((multivariate_normal.med(mu=mu, sigma=sigma), ), _custom_likelihood, (y, x), multivariate_normal, None)
        ]

        for test in tests:
            _run_test(*test)


if __name__ == '__main__':
    unittest.main()
