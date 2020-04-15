import unittest
import time
import numpy as np
from probpy.distributions import normal
import numba
from probpy.density import UCKD, RCKD, URBK
from probpy.sampling import fast_metropolis_hastings
from probpy.search import search_posterior_estimation
from probpy.distributions import normal, exponential, jit
from probpy.learn.posterior.common import jit_log_probabilities


def distribution(x):
    return 0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2)


def log_distribution(x):
    return np.log(0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2))


class AutomaticDensityTest(unittest.TestCase):
    def test_running_uckd(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(5000, distribution, initial=np.random.rand(10, 1), energy=1.0)
        print("making samples", time.time() - timestamp)

        density = UCKD(variance=5.0)
        density.fit(samples)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        y = density.p(x)
        y = y / (y.sum() / (n / (ub - lb)))

        delta = (n / (ub - lb))
        self.assertAlmostEqual(y.sum() / delta, 1, delta=0.1)

        fast_p = density.get_fast_p()

        fast_p(x)  # is slower than normal p.. but numba need numba functions

    def test_running_rckd(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(5000, distribution, initial=np.random.rand(50, 1), energy=1.0)
        print("making samples", time.time() - timestamp)

        density = RCKD(variance=5.0, error=0.001, verbose=True)
        timestamp = time.time()
        density.fit(samples)
        print("fitting samples", time.time() - timestamp)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        print("x", len(x))
        y = density.p(x)

        delta = (n / (ub - lb))
        self.assertAlmostEqual(y.sum() / delta, 1, delta=0.5)

        fast_p = density.get_fast_p()

        fast_p(x)  # is slower than normal p.. but numba need numba functions

    def test_running_urbk(self):
        prior_rv = normal.med(mu=0.5, sigma=1.0)
        n = normal.fast_p

        prior = jit.jit_probability(prior_rv)

        @numba.jit(fastmath=True, nopython=True, forceobj=False)
        def likelihood(y, w):
            return n(y - w, mu=0.0, sigma=1.0)

        data = normal.sample(mu=3.0, sigma=1.0, size=100)

        log_likelihood, log_prior = jit_log_probabilities((data,), likelihood, prior)

        samples, densities = search_posterior_estimation(
            size=300, log_likelihood=log_likelihood,
            log_prior=log_prior,
            initial=prior_rv.sample(size=10),
            energy=0.1,
            volume=100
        )

        density = URBK(variance=5.0, verbose=True)
        density.fit(samples, densities)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        y = density.p(x)
        self.assertEqual(y.size, 2000)

        fast_p = density.get_fast_p()

        fast_p(x)  # is slower than normal p.. but numba need numba functions


if __name__ == '__main__':
    unittest.main()
