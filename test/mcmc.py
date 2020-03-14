import unittest
from probpy.distributions import normal
from probpy.distributions import uniform
from probpy.mcmc import metropolis_hastings, metropolis
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


class TestMCMC(unittest.TestCase):
    def test_metropolis_hastings(self):
        pdf = lambda x: 0.3333 * normal.p(0, 1, x) + 0.3333 * normal.p(6, 3, x) + 0.3333 * normal.p(-6, 0.5, x)
        x = np.linspace(-10, 10, 1000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.lineplot(x, pdf(x))

        plt.subplot(2, 1, 2)
        proposal = lambda x, y: normal.p(x, 1, y)
        sampler = lambda x: normal.sample(x, 1)
        samples = metropolis_hastings(pdf, proposal, sampler, shape=(30000, 1))[100::3]

        sb.distplot(samples)
        plt.show()

    def test_metropolis(self):
        pdf = lambda x: normal.p(0, 1, x) + normal.p(6, 3, x) + normal.p(-6, 0.5, x)
        x = np.linspace(-10, 10, 1000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title("True", fontsize=18)
        sb.lineplot(x, pdf(x))

        plt.subplot(2, 1, 2)
        plt.title("Estimated with 10000 samples", fontsize=18)
        samples = metropolis(size=10000, pdf=pdf, proposal=normal.freeze(mu=0, sigma=10), M=30.0)
        samples = samples[1000:]

        sb.distplot(samples)
        plt.tight_layout()
        plt.savefig("../images/metropolis.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
