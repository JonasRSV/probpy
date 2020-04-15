import unittest
import probpy as pp
import numpy as np
import numba
from probpy.search import search_posterior_estimation
from probpy.distributions import normal, exponential, jit
from probpy.learn.posterior.common import jit_log_probabilities


class TestSearch(unittest.TestCase):
    def test_parameter_posterior_search(self):
        prior_rv = normal.med(mu=0.5, sigma=1.0)
        n = normal.fast_p

        prior = jit.jit_probability(prior_rv)

        @numba.jit(fastmath=True, nopython=True, forceobj=False)
        def likelihood(y, w):
            return n(y - w, mu=0.0, sigma=1.0)

        data = normal.sample(mu=3.0, sigma=1.0, size=100)

        log_likelihood, log_prior = jit_log_probabilities((data,), likelihood, prior)

        points, densities = search_posterior_estimation(
            size=1000, log_likelihood=log_likelihood,
            log_prior=log_prior,
            initial=prior_rv.sample(size=10),
            energy=0.1,
            volume=1000
        )

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
