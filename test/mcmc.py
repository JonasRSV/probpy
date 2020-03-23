import unittest
from probpy.distributions import normal
from probpy.distributions import uniform
from probpy.mcmc import metropolis_hastings, metropolis
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


class TestMCMC(unittest.TestCase):
    def test_metropolis_hastings(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
        x = np.linspace(-10, 10, 1000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.lineplot(x, pdf(x))

        plt.subplot(2, 1, 2)
        samples = metropolis_hastings(50000, pdf, normal.freeze(sigma=1.0), initial=-5)
        samples = samples[10000:]

        sb.distplot(samples)
        plt.xlim([-10, 10])
        plt.savefig("../images/metropolis-hastings.png", bbox_inches="tight")
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
        samples = metropolis(size=100000, pdf=pdf, proposal=normal.freeze(mu=0, sigma=10), M=30.0)
        samples = samples[1000:]

        sb.distplot(samples)
        plt.tight_layout()
        plt.xlim([-10, 10])
        plt.savefig("../images/metropolis.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
