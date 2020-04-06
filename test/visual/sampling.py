import unittest
from probpy.distributions import normal
from probpy.sampling import metropolis_hastings, metropolis
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


class VisualTestMCMC(unittest.TestCase):
    def test_metropolis_hastings_visual(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
        x = np.linspace(-10, 10, 1000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.lineplot(x, pdf(x))

        plt.subplot(2, 1, 2)
        samples = metropolis_hastings(50000, pdf, normal.med(sigma=1.0), initial=-5)
        samples = samples[1000:]

        sb.distplot(samples)
        plt.xlim([-10, 10])
        plt.savefig("../../images/metropolis-hastings.png", bbox_inches="tight")
        plt.show()

    def test_metropolis_visual(self):
        pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
        x = np.linspace(-10, 10, 1000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title("True", fontsize=18)
        sb.lineplot(x, pdf(x))

        plt.subplot(2, 1, 2)
        plt.title("Estimated with 1000000 samples", fontsize=18)
        samples = metropolis(size=1000000, pdf=pdf, proposal=normal.med(mu=0, sigma=10), M=30.0)
        sb.distplot(samples)
        plt.tight_layout()
        plt.xlim([-10, 10])
        plt.savefig("../../images/metropolis.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
