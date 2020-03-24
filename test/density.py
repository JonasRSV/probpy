import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from probpy.distributions import normal
from probpy.density import UCKD, RCKD
from probpy.mcmc import fast_metropolis_hastings


def distribution(x):
    return 0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2)


class DensityTest(unittest.TestCase):
    def test_uckd_by_inspection(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(50000, distribution, initial=0.0, energy=1.0)
        print("making samples", time.time() - timestamp)

        density = UCKD(variance=5.0)
        density.fit(samples)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        timestamp = time.time()
        y = density.p(x)
        print("predicting samples", time.time() - timestamp)
        y = y / (y.sum() / (n / (ub - lb)))

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="UCKD")
        plt.plot(x, distribution(x), label="PDF sampled from")
        sb.distplot(samples, label="Histogram of samples")
        plt.tight_layout()
        plt.legend(fontsize=16)
        plt.savefig("../images/uckd.png", bbox_inches="tight")
        plt.show()

    def test_rckd_by_inspection(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(50000, distribution, initial=0.0, energy=1.0).reshape(-1, 1)
        print("making samples", time.time() - timestamp)

        density = RCKD(variance=5.0, error=1, verbose=True)
        timestamp = time.time()
        density.fit(samples)
        print("fitting samples", time.time() - timestamp)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        timestamp = time.time()
        y = density.p(x)
        print("y", y)
        print("predicting samples", time.time() - timestamp)

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="RCKD")
        plt.plot(x, distribution(x), label="PDF sampled from")
        sb.distplot(samples, label="Histogram of samples")
        plt.legend(fontsize=16)
        plt.savefig("../images/rckd.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
