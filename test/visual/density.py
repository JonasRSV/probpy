import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from probpy.distributions import normal
from probpy.density import UCKD, RCKD, URBK
from probpy.sampling import fast_metropolis_hastings, ga_posterior_estimation


def distribution(x):
    return 0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2)


def log_distribution(x):
    return np.log(0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2))


class VisualDensityTest(unittest.TestCase):
    def test_uckd_by_inspection_visual(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(50000, distribution, initial=np.random.rand(10, 1), energy=1.0)
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
        plt.savefig("../../images/uckd.png", bbox_inches="tight")
        plt.show()

    def test_rckd_by_inspection_visual(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(50000, distribution, initial=np.random.rand(10, 1), energy=1.0)
        print("making samples", time.time() - timestamp)

        density = RCKD(variance=5.0, error=0.01, verbose=True)
        timestamp = time.time()
        density.fit(samples)
        print("fitting samples", time.time() - timestamp)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        timestamp = time.time()
        y = density.p(x)
        print("predicting samples", time.time() - timestamp)

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="RCKD")
        plt.plot(x, distribution(x), label="PDF sampled from")
        sb.distplot(samples, label="Histogram of samples")
        plt.legend(fontsize=16)
        plt.savefig("../../images/rckd.png", bbox_inches="tight")
        plt.show()

    def test_urbk_by_inspection_visual(self):
        timestamp = time.time()

        log_priors = [lambda x: np.log(normal.p(x, mu=0.0, sigma=10.0))]

        samples, densities = ga_posterior_estimation(50000,
                                                     log_distribution,
                                                     log_priors,
                                                     initial=np.random.rand(1, 10) * 10,
                                                     energies=np.repeat(1.0, 10))
        print("making samples", time.time() - timestamp)

        density = URBK(variance=0.5, use_cl=True, verbose=True)
        timestamp = time.time()
        density.fit(samples, densities)
        print("fitting samples", time.time() - timestamp)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        timestamp = time.time()
        y = density.p(x)
        y = (y / y.sum()) * (n / (ub - lb))
        print("predicting samples", time.time() - timestamp)
        print("y", y)
        print("y.sum", y.sum())

        print(y.shape)
        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label="URBK")
        plt.plot(x, distribution(x), label="PDF sampled from")
        sb.distplot(samples, label="Histogram of samples")
        plt.legend(fontsize=16)
        plt.savefig("../../images/urbk.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
