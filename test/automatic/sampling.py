import unittest
from probpy.distributions import normal, exponential, jit
from probpy.learn.posterior.common import jit_log_probabilities
from probpy.sampling import (metropolis_hastings,
                             metropolis,
                             fast_metropolis_hastings_log_space,
                             fast_metropolis_hastings,
                             fast_metropolis_hastings_log_space_parameter_posterior_estimation)
import numpy as np
import numba


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
        prior_rv = normal.med(mu=0.5, sigma=1.0)
        n = normal.fast_p

        prior = jit.jit_probability(prior_rv)

        @numba.jit(fastmath=True, nopython=True, forceobj=False)
        def likelihood(y, w):
            return n(y - w, mu=0.0, sigma=1.0)

        data = normal.sample(mu=3.0, sigma=1.0, size=100)

        log_likelihood, log_prior = jit_log_probabilities((data,), likelihood, prior)

        result = fast_metropolis_hastings_log_space_parameter_posterior_estimation(
            size=2000, log_likelihood=log_likelihood,
            log_prior=log_prior,
            initial=prior_rv.sample(size=10),
            energy=0.1
        )

        self.assertAlmostEqual(result.mean(), 3.0, delta=1.0)

    def test_metropolis(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
        samples = metropolis(size=100000, pdf=pdf, proposal=normal.med(mu=0, sigma=10), M=30.0)
        self.assertAlmostEqual(samples.mean(), 0.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
