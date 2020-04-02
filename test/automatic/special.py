import unittest
import numpy as np
from scipy.special import gamma as gamma_
from scipy.special import beta as beta_
from probpy.special import gamma
from probpy.special import beta


class SpecialTest(unittest.TestCase):
    def test_running_gamma(self):
        x = np.linspace(0.1, 10)
        y = gamma(x)
        z = gamma_(x)

        self.assertAlmostEqual(np.square(y - z).mean(), 0.0, delta=1e-2)

    def test_running_beta(self):
        points = 100
        x = np.linspace(0.1, 5, points)
        y = np.linspace(0.1, 5, points)

        X, Y = np.meshgrid(x, y)

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        i_beta = beta(X, Y)
        scipy_beta = beta_(X, Y)

        i_beta = i_beta.reshape(points, points)
        scipy_beta = scipy_beta.reshape(points, points)

        self.assertAlmostEqual(np.abs(i_beta - scipy_beta).mean(), 0.0, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
