import unittest
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
import numpy as np


class PosteriorTest(unittest.TestCase):
    def test_conjugates(self):
        def _run_test(prior=None, likelihood=None, data=None, correct=None):
            posterior = parameter_posterior(data, likelihood=likelihood, priors=prior, size=1000)

            if correct is not None:
                pass  # TODO

        mu_prior, sigma_prior = np.ones(2), np.eye(2)
        probabilities = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        x = np.linspace(0, 1, 10)
        y = x + 1 + normal.sample(mu=0.0, sigma=1e-1, size=10)
        tests = [
            {
                "prior": normal.med(mu=1.0, sigma=1.0),
                "likelihood": normal.med(sigma=2.0),
                "data": normal.med(mu=-2.0, sigma=2.0).sample(size=200),
                "correct": None
            },
            {
                "prior": normal_inverse_gamma.med(mu=1.0, lam=2.0, a=3.0, b=3.0),
                "likelihood": normal.med(),
                "data": normal.med(mu=-2.0, sigma=2.0).sample(size=200),
                "correct": None
            },
            {
                "prior": multivariate_normal.med(mu=mu_prior, sigma=sigma_prior),
                "likelihood": multivariate_normal.med(sigma=sigma_prior),
                "data": multivariate_normal.med(mu=mu_prior, sigma=sigma_prior).sample(size=200),
                "correct": None
            },
            {
                "prior": beta.med(a=1.0, b=3.0),
                "likelihood": bernoulli.med(),
                "data": bernoulli.med(probability=0.6).sample(size=200),
                "correct": None
            },
            {
                "prior": dirichlet.med(alpha=np.ones(5)),
                "likelihood": categorical.med(dim=5),
                "data": categorical.med(probabilities=probabilities).sample(size=200),
                "correct": None
            },
            {
                "prior": gamma.med(a=9, b=2),
                "likelihood": exponential.med(),
                "data": exponential.med(lam=1.0).sample(size=200),
                "correct": None
            },
            {
                "prior": beta.med(a=6.0, b=3.0),
                "likelihood": binomial.med(n=5),
                "data": binomial.med(n=5, probability=0.7).sample(size=200),
                "correct": None
            },
            {
                "prior": dirichlet.med(alpha=np.ones(3)),
                "likelihood": multinomial.med(n=3),
                "data": multinomial.med(n=3, probabilities=np.ones(3) / 3).sample(size=100),
                "correct": None
            },
            {
                "prior": gamma.med(a=9.0, b=2.0),
                "likelihood": poisson.med(),
                "data": poisson.med(lam=2).sample(size=200),
                "correct": None
            },
            {
                "prior": beta.med(a=6.0, b=3.0),
                "likelihood": geometric.med(),
                "data": geometric.med(probability=0.7).sample(size=200),
                "correct": None
            },
            {
                "prior": multivariate_normal.med(mu=mu_prior, sigma=sigma_prior),
                "likelihood": unilinear.med(sigma=0.1),
                "data": (y, x),
                "correct": None
            },
            {
                "prior": multivariate_normal.med(mu=mu_prior, sigma=sigma_prior),
                "likelihood": unilinear.med(sigma=0.1),
                "data": (1.0, 2.0),
                "correct": None
            },
            {
                "prior": normal.med(mu=1.0, sigma=1.0),
                "likelihood": normal.med(sigma=2.0),
                "data": 2.0,
                "correct": None
            },
            {
                "prior": normal_inverse_gamma.med(mu=1.0, lam=2.0, a=3.0, b=3.0),
                "likelihood": normal.med(),
                "data": 2.0,
                "correct": None
            },
            {
                "prior": multivariate_normal.med(mu=mu_prior, sigma=sigma_prior),
                "likelihood": multivariate_normal.med(sigma=sigma_prior),
                "data": np.zeros(2),
                "correct": None
            },
            {
                "prior": beta.med(a=1.0, b=3.0),
                "likelihood": bernoulli.med(),
                "data": 1,
                "correct": None
            },
            {
                "prior": dirichlet.med(alpha=np.ones(5)),
                "likelihood": categorical.med(dim=5),
                "data": 3,
                "correct": None
            },
            {
                "prior": gamma.med(a=9, b=2),
                "likelihood": exponential.med(),
                "data": 5.1,
                "correct": None
            },
            {
                "prior": beta.med(a=6.0, b=3.0),
                "likelihood": binomial.med(n=5),
                "data": 4,
                "correct": None
            },
            {
                "prior": dirichlet.med(alpha=np.ones(3)),
                "likelihood": multinomial.med(n=3),
                "data": [1, 1, 1],
                "correct": None
            },
            {
                "prior": gamma.med(a=9.0, b=2.0),
                "likelihood": poisson.med(),
                "data": 7,
                "correct": None
            },
            {
                "prior": beta.med(a=6.0, b=3.0),
                "likelihood": geometric.med(),
                "data": 6,
                "correct": None
            },
        ]

        for test in tests:
            _run_test(**test)

    def test_mcmc(self):

        def _run_test(priors=None, likelihood=None, data=None, correct=None):
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

        tests = (
            {
                "priors": exponential.med(lam=1.0),
                "likelihood": normal.med(sigma=2.0),
                "data": normal.sample(mu=3.0, sigma=2.0, size=200),
                "correct": None
            },
            {
                "priors": (normal.med(mu=3.0, sigma=2.0), exponential.med(lam=1.0)),
                "likelihood": normal.med(),
                "data": normal.sample(mu=5.0, sigma=2.0, size=300),
                "correct": None
            },
            {
                "priors": multivariate_uniform.med(a=a, b=b),
                "likelihood": multivariate_normal.med(sigma=sigma_prior),
                "data": multivariate_normal.sample(mu=mu_prior, sigma=sigma_prior, size=100),
                "correct": None
            },
            {
                "priors": (multivariate_normal.med(mu=mu_prior, sigma=sigma_prior), exponential.med(lam=1.0)),
                "likelihood": unilinear.med(),
                "data": (y, x),
                "correct": None
            },
            {
                "priors": multivariate_normal.med(mu=mu_prior, sigma=sigma_prior),
                "likelihood": _custom_likelihood,
                "data": (logistic_y, logistic_x),
                "correct": None
            })

        for test in tests:
            _run_test(**test)

    def test_mcmc_moment_matching(self):
        def _run_test(priors=None, likelihood=None, data=None, match=None, correct=None):
            posterior = parameter_posterior(data, likelihood=likelihood,
                                            priors=priors, size=3000, batch=5, match_moments_for=match)

            if correct is not None:
                pass  # TODO

            if hasattr(posterior, "__iter__"):
                for v in posterior:
                    print(v)
            else:
                print(posterior)
            print("\n\n")

        def _custom_likelihood(y, x, w):
            result = []
            for _w in w: result.append(normal.p(y - x * _w[0] - _w[1], mu=0.0, sigma=0.3))
            return np.array(result)

        a, b = np.ones(2) * -2, np.ones(2) * 2
        sigma = np.eye(2)
        mu = np.zeros(2)
        x = np.linspace(-1, 1, 100)
        y = x * 2 + 0.5 + normal.sample(mu=0.0, sigma=0.3, size=100)

        tests = [
            {
                "priors": exponential.med(lam=0.6, ),
                "likelihood": normal.med(sigma=1.0),
                "data": normal.sample(mu=3.0, sigma=2.0, size=200),
                "match": normal,
                "correct": None
            },
            {
                "priors": multivariate_uniform.med(a=a, b=b),
                "likelihood": multivariate_normal.med(sigma=sigma),
                "data": multivariate_normal.sample(mu=mu, sigma=sigma, size=500),
                "match": multivariate_normal,
                "correct": None,
            },
            {
                "priors": multivariate_normal.med(mu=mu, sigma=sigma),
                "likelihood": _custom_likelihood,
                "data": (y, x),
                "match": multivariate_normal,
                "correct": None
            },
            {
                "priors": (multivariate_normal.med(mu=np.ones(2), sigma=np.eye(2)), exponential.med(lam=0.5)),
                "likelihood": unilinear.med(),
                "data": (y, x),
                "match": (multivariate_normal, exponential),
                "correct": None
            }
        ]

        for test in tests:
            _run_test(**test)


if __name__ == '__main__':
    unittest.main()
