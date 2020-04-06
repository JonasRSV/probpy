import unittest
import numpy as np
from probpy.distributions import *
import itertools


class TestDistributions(unittest.TestCase):

    def test_first_two_moments(self):

        probabilities = np.array([0.3, 0.5, 0.2])
        alphas = np.array([1.0, 2.0, 1.0])
        norm_alpha = alphas / alphas.sum()
        tests = [
            (normal.med(mu=0.0, sigma=2.0), (0.0, 2.0)),
            (multivariate_normal.med(mu=np.zeros(2), sigma=np.eye(2)), (np.zeros(2), np.eye(2))),
            (uniform.med(a=0, b=1), (0.5, np.square(0.5) / 12)),
            (multivariate_uniform.med(a=np.zeros(2), b=np.ones(2)),
             (np.ones(2) * 0.5, np.square(np.ones(2) * 0.5) / 12)),
            (bernoulli.med(probability=0.5), (0.5, 0.25)),
            (categorical.med(probabilities=probabilities), (probabilities, probabilities * (1 - probabilities))),
            (dirichlet.med(alpha=alphas), (norm_alpha, norm_alpha * (1 - norm_alpha) / (alphas.sum() + 1))),
            (beta.med(a=1, b=1), (1 / 2, 1 / (np.square(2) * 3))),
            (exponential.med(lam=2), (1 / 2, 1 / np.square(2))),
            (binomial.med(probability=0.7, n=20), (0.7 * 20, None)),
            (multinomial.med(probabilities=probabilities, n=10), (probabilities * 10, None)),
            (gamma.med(a=9.0, b=2.0), (9.0 / 2.0, None)),
            (normal_inverse_gamma.med(mu=2.0, lam=1.0, a=2.0, b=2.0), ([2.0, 2.0], None)),
            (geometric.med(probability=0.7), (1 / 0.7, (1 - 0.7) / (0.7 * 0.7))),
            (poisson.med(lam=4.0), (4.0, 4.0)),
            (hypergeometric.med(N=10, K=5, n=3), (3 * 5 / 10,
                                                  3 * (5 / 10) * (5 / 10) * 7 / 9))

        ]

        def _test_m1(samples, m1, error: str):
            if samples.ndim == 1:
                m1_estimate = samples.mean()
            else:
                m1_estimate = samples.mean(axis=0)

            if m1_estimate.ndim == 0: self.assertAlmostEqual(m1_estimate, m1, delta=0.1, msg=error)
            if m1_estimate.ndim == 1:
                for i in range(m1_estimate.size): self.assertAlmostEqual(m1_estimate[i], m1[i], delta=0.1, msg=error)

        def _test_m2(samples, m2, error: str):
            if samples.ndim == 1: m2_estimate = samples.var()
            if samples.ndim == 2: m2_estimate = np.cov(samples, rowvar=False)

            if samples.ndim == 2 and m2.ndim == 1:
                m2_estimate = np.diag(m2_estimate)

            if m2_estimate.ndim == 0: self.assertAlmostEqual(m2_estimate, m2, delta=0.1, msg=error)
            if m2_estimate.ndim == 1:
                x = m2_estimate.size
                for i in range(x):
                    self.assertAlmostEqual(m2_estimate[i], m2[i], delta=0.1, msg=error)

            if m2_estimate.ndim == 2:
                x, y = m2_estimate.shape
                for i in range(x):
                    for j in range(y):
                        self.assertAlmostEqual(m2_estimate[i, j], m2[i, j], delta=0.1, msg=error)

        for distribution, (m_1, m_2) in tests:
            samples = distribution.sample(size=100000)

            if m_1 is not None: _test_m1(samples, m_1, f"{distribution.cls} failed")
            if m_2 is not None: _test_m2(samples, m_2, f"{distribution.cls} failed")

    def test_med(self):
        def _execute_test(distribution, parameters, names, error: str):
            s, p = [], []
            for test_case in itertools.product([0, 1], repeat=len(parameters)):
                kwargs, args = {}, []
                for i, pick in enumerate(test_case):
                    if pick == 1:
                        kwargs[names[i]] = parameters[i]
                    else:
                        args.append(parameters[i])

                rv = distribution.med(**kwargs)
                _s = rv.sample(*args, size=100000)
                _p = rv.p(_s, *args)

                s.append(_s.mean())
                p.append(_p.mean())

            s = np.array(s)
            p = np.array(p)

            np.testing.assert_almost_equal(s - s[0], np.zeros_like(s), decimal=1, err_msg=error)
            np.testing.assert_almost_equal(p - p[0], np.zeros_like(p), decimal=1, err_msg=error)

        tests = [
            (normal, (2.0, 1.0), (normal.mu, normal.sigma)),
            (multivariate_normal, (np.zeros(2), np.eye(2)), (multivariate_normal.mu, multivariate_normal.sigma)),
            (uniform, (0.0, 1.0), (uniform.a, uniform.b)),
            (multivariate_uniform, (np.zeros(2), np.ones(2)), (multivariate_uniform.a, multivariate_uniform.b)),
            (bernoulli, (0.5,), (bernoulli.probability,)),
            (beta, (2.0, 1.0), (beta.a, beta.b)),
            (binomial, (3, 0.5), (binomial.n, binomial.probability)),
            (categorical, (np.ones(2) * 0.5,), (categorical.probabilities,)),
            (dirichlet, (np.ones(2) * 2,), (dirichlet.alpha,)),
            (exponential, (1.0,), (exponential.lam,)),
            (multinomial, (10, np.ones(4) * 0.25,), (multinomial.n, multinomial.probabilities)),
            (gamma, (2.0, 1.0), (gamma.a, gamma.b)),
            (normal_inverse_gamma, (2, 1, 2, 2), (normal_inverse_gamma.mu, normal_inverse_gamma.lam,
                                                  normal_inverse_gamma.a, normal_inverse_gamma.b)),
            (geometric, (0.5,), (geometric.probability,)),
            (poisson, (2.0,), (poisson.lam,)),
            (hypergeometric, (10, 5, 3), (hypergeometric.N, hypergeometric.K, hypergeometric.n)),
            (gaussian_process, (
                np.array([0.0, 0.5]),
                lambda x: 0,
                lambda x, y: np.exp(-1.0 * np.square(x - y)),
                np.linspace(0, 1, 5),
                np.random.rand(5)
            ),
             (
                 gaussian_process.x,
                 gaussian_process.mu,
                 gaussian_process.sigma,
                 gaussian_process.X,
                 gaussian_process.Y
             )),
            (unilinear,
             (
                 np.linspace(0, 1, 100),
                 np.ones(2),
                 0.1
             ),
             (
                 unilinear.x,
                 unilinear.variables,
                 unilinear.sigma
             ))
        ]

        for distribution, parameters, names in tests:
            _execute_test(distribution, parameters, names, f"{distribution.__name__} failed")

    def test_p(self):
        def _execute_test(distribution=None, data=None, correct=None):
            for d, c in zip(data, correct):
                result = distribution.p(d)

                if c is not None:
                    pass  # TODO

        tests = [
            {
                "distribution": normal.med(mu=0.0, sigma=1.0),
                "data": [1.0, [1.0, 0.5], np.ones(2)],
                "correct": [None, None, None]
            },
            {
                "distribution": multivariate_normal.med(mu=np.ones(2), sigma=np.eye(2)),
                "data": [[0.0, 0.0], np.ones(2), np.ones((2, 2)), [[0.1, 0.2], [0.5, 0.2]]],
                "correct": [None, None, None, None]
            },
            {
                "distribution": uniform.med(a=0.0, b=1.0),
                "data": [0.5, [0.5], np.ones(1), np.ones(10) * 0.5],
                "correct": [None, None, None, None]
            },
            {
                "distribution": multivariate_uniform.med(a=np.zeros(2), b=np.ones(2)),
                "data": [[0.0, 0.0], np.zeros(2), [[0.0, 1.0], [0.5, 1.0]], np.random.rand(2, 2)],
                "correct": [None, None, None, None]
            },
            {
                "distribution": bernoulli.med(probability=0.6),
                "data": [0.5, [0.5], [0.5, 0.5]],
                "correct": [None, None, None, None]
            },
            {
                "distribution": categorical.med(probabilities=np.ones(4) / 4),
                "data": [3, [1, 2], np.eye(4)[[1, 2]], [[0, 0, 0, 1]]],
                "correct": [None, None, None, None]
            },
            {
                "distribution": dirichlet.med(alpha=np.ones(4)),
                "data": [np.ones(4) / 4, [0.25, 0.25, 0.25, 0.25], [[0.25, 0.25, 0.25, 0.25]], np.ones((1, 4))],
                "correct": [None, None, None, None]
            },
            {
                "distribution": beta.med(a=1.0, b=2.0),
                "data": [0.5, [0.5], [0.5, 0.5], np.random.rand(), np.random.rand(3)],
                "correct": [None, None, None, None]
            },
            {
                "distribution": exponential.med(lam=1.0),
                "data": [0.5, [0.5], [0.5, 0.5], np.random.rand(), np.random.rand(3)],
                "correct": [None, None, None, None]
            },
            {
                "distribution": binomial.med(n=2, probability=0.5),
                "data": [1, [1], [0, 0, 1], np.ones(2)],
                "correct": [None, None, None, None]
            },
            {
                "distribution": multinomial.med(n=3, probabilities=np.ones(3) / 3),
                "data": [[1, 1, 1], np.ones(3), [[1, 1, 1]], np.ones((2, 3))],
                "correct": [None, None, None, None]
            },
            {
                "distribution": gamma.med(a=1.0, b=2.0),
                "data": [0.5, [0.5], [0.5, 0.5], np.random.rand(), np.random.rand(3)],
                "correct": [None, None, None, None]
            },
            {
                "distribution": normal_inverse_gamma.med(mu=2.0, lam=1.0, a=1.0, b=2.0),
                "data": [[0.5, 1.0], [[0.5, 1.0]], np.ones(2), np.ones((2, 2))],
                "correct": [None, None, None, None]
            },
            {
                "distribution": geometric.med(probability=0.7),
                "data": [5, [5], np.ones(1) * 5, np.ones(5) * 5],
                "correct": [None, None, None, None]
            },
            {
                "distribution": hypergeometric.med(N=3, K=2, n=1),
                "data": [1.0, [1.0], [1.0, 1.0], np.ones(2)],
                "correct": [None, None, None, None]
            },
            {
                "distribution": poisson.med(lam=2.0),
                "data": [1.0, [1.0], [1.0, 1.0], np.ones(2)],
                "correct": [None, None, None, None]
            }
        ]

        for test in tests:
            _execute_test(**test)


if __name__ == '__main__':
    unittest.main()
