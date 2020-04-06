import unittest
from probpy.distributions import normal, exponential
from probpy.sampling import (metropolis_hastings,
                         metropolis,
                         fast_metropolis_hastings_log_space,
                         fast_metropolis_hastings,
                         fast_metropolis_hastings_log_space_parameter_posterior_estimation)
import numpy as np


class TestSampling(unittest.TestCase):
    def test_metropolis_hastings(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3)
        log_pdf = lambda x: np.log(normal.p(x, 0, 1) + normal.p(x, 6, 3))

        samples = metropolis_hastings(50000, pdf, normal.med(sigma=1.0), initial=-5)
        fast_samples = fast_metropolis_hastings(500000, pdf, initial=(np.random.rand(100) - 0.5) * 10.0, energy=1.0)
        log_fast_samples = fast_metropolis_hastings_log_space(500000, log_pdf,
                                                              initial=(np.random.rand(100) - 0.5) * 10.0, energy=1.0)

        samples_mean = samples[10000:].mean()
        fast_samples_mean = fast_samples[100000:].mean()
        log_fast_samples = log_fast_samples[100000:].mean()

        for i in [samples_mean, fast_samples_mean, log_fast_samples]:
            for j in [samples_mean, fast_samples_mean, log_fast_samples]:
                self.assertAlmostEqual(i, j, delta=0.5)

    def test_parameter_posterior_mcmc(self):
        prior_mu = normal.med(mu=0.5, sigma=1.0)
        prior_sigma = exponential.med(lam=1.0)
        likelihood = normal.med()

        data = normal.sample(mu=3.0, sigma=1.0, size=100)

        def log_likelihood(*args):
            ll = np.log(likelihood.p(data, *args)).sum(axis=1)
            return np.nan_to_num(ll, copy=False, nan=-10000.0)

        def log_prior_mu(b): return np.log(prior_mu.p(b))
        def log_prior_sigma(b): return np.log(prior_sigma.p(b))

        result = fast_metropolis_hastings_log_space_parameter_posterior_estimation(
            size=20000, log_likelihood=log_likelihood,
            log_priors=[log_prior_mu, log_prior_sigma],
            initial=[prior_mu.sample(size=10), prior_sigma.sample(size=10)],
            energies=(0.1, 0.1)
        )

        correct = [3.0, 1.0]
        for res, corr in zip(result, correct):
            self.assertAlmostEqual(res.mean(), corr, delta=0.5)

    def test_metropolis(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
        samples = metropolis(size=100000, pdf=pdf, proposal=normal.med(mu=0, sigma=10), M=30.0)
        self.assertAlmostEqual(samples.mean(), 0.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
