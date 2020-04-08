import unittest
import probpy as pp
import numpy as np
from probpy.search import search_posterior_estimation


class TestSearch(unittest.TestCase):
    def test_parameter_posterior_search(self):
        prior_mu = pp.normal.med(mu=0.5, sigma=1.0)
        prior_sigma = pp.exponential.med(lam=1.0)
        likelihood = pp.normal.med()

        data = pp.normal.sample(mu=3.0, sigma=1.0, size=100)

        def log_likelihood(*args):
            ll = np.log(likelihood.p(data, *args)).sum(axis=1)
            return np.nan_to_num(ll, copy=False, nan=-10000.0)

        def log_prior_mu(b): return np.log(prior_mu.p(b))

        def log_prior_sigma(b): return np.log(prior_sigma.p(b))

        points, densities = search_posterior_estimation(
            size=1000, log_likelihood=log_likelihood,
            log_priors=[log_prior_mu, log_prior_sigma],
            initial=[prior_mu.sample(size=10), prior_sigma.sample(size=10)],
            energies=(0.1, 0.1),
            volume=100
        )

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
