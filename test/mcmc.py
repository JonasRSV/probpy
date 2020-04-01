import unittest
from probpy.distributions import normal
from probpy.mcmc import metropolis_hastings, metropolis, fast_metropolis_hastings_log_space, fast_metropolis_hastings
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import time
import numba


class TestMCMC(unittest.TestCase):
    def test_metropolis_hastings(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
        x = np.linspace(-10, 10, 1000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.lineplot(x, pdf(x))

        plt.subplot(2, 1, 2)
        timestamp = time.time()
        samples = metropolis_hastings(50000, pdf, normal.med(sigma=1.0), initial=-5)
        print(f"MH took {time.time() - timestamp}")

        timestamp = time.time()
        fast_samples = fast_metropolis_hastings(500000, pdf, initial=(np.random.rand(100) - 0.5) * 10.0, energy=1.0)
        print(f"fast MH took {time.time() - timestamp}")

        #timestamp = time.time()
        #fast_samples = fast_metropolis_hastings_log_space(400000, pdf, initial=(np.random.rand(1000) - 0.5) * 10.0, energy=1.0)
        #print(f"fast MH took {time.time() - timestamp}")

        samples = fast_samples[100000:]
        # samples = samples[10000:]

        sb.distplot(samples)
        plt.xlim([-10, 10])
        #plt.savefig("../images/metropolis-hastings.png", bbox_inches="tight")
        plt.show()

    def test_metropolis(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
        x = np.linspace(-10, 10, 1000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title("True", fontsize=18)
        sb.lineplot(x, pdf(x))

        plt.subplot(2, 1, 2)
        plt.title("Estimated with 100000 samples", fontsize=18)
        timestamp = time.time()
        samples = metropolis(size=1000000, pdf=pdf, proposal=normal.med(mu=0, sigma=10), M=30.0)
        print(time.time() - timestamp)

        samples = samples[1000:]

        sb.distplot(samples)
        plt.tight_layout()
        plt.xlim([-10, 10])
        plt.savefig("../images/metropolis.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
