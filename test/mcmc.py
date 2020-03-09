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
        sb.lineplot(x, pdf(x))

        a, b = np.array(-10), np.array(10)

        plt.subplot(2, 1, 2)
        #proposal = lambda x: uniform.p(a, b, x)
        #sampler = lambda: uniform.sample(a, b)
        proposal = lambda x: normal.p(0, 10, x)
        sampler = lambda: normal.sample(0, 10)

        samples = metropolis(pdf, proposal, sampler, M=30.0, shape=(10000, 1))[1000:]

        sb.distplot(samples)
        plt.show()


if __name__ == '__main__':
    unittest.main()
