import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from probpy.distributions import normal
from probpy.density import UCKD, RCKD
from probpy.sampling import fast_metropolis_hastings


def distribution(x):
    return 0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2)


class AutomaticDensityTest(unittest.TestCase):
    def test_running_uckd(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(50000, distribution, initial=np.random.rand(10, 1), energy=1.0)
        print("making samples", time.time() - timestamp)

        density = UCKD(variance=5.0)
        density.fit(samples)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        y = density.p(x)
        y = y / (y.sum() / (n / (ub - lb)))

        delta = (n / (ub - lb))
        self.assertAlmostEqual(y.sum() / delta, 1, delta=0.1)

    def test_running_rckd(self):
        timestamp = time.time()
        samples = fast_metropolis_hastings(50000, distribution, initial=np.random.rand(50, 1), energy=1.0)
        print("making samples", time.time() - timestamp)

        density = RCKD(variance=5.0, error=0.001, verbose=True)
        timestamp = time.time()
        density.fit(samples)
        print("fitting samples", time.time() - timestamp)

        lb, ub = -6, 6
        n = 2000

        x = np.linspace(lb, ub, n)
        y = density.p(x)

        delta = (n / (ub - lb))
        self.assertAlmostEqual(y.sum() / delta, 1, delta=0.1)



if __name__ == '__main__':
    unittest.main()
