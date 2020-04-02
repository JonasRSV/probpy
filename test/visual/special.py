import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.special import gamma as gamma_
from scipy.special import beta as beta_
from probpy.special import gamma
from probpy.special import beta


class VisualSpecialTest(unittest.TestCase):
    def test_gamma_visual(self):
        x = np.linspace(0.1, 10)
        y = gamma(x)
        z = gamma_(x)

        plt.figure(figsize=(10, 6))
        sb.lineplot(x, y, label="Local")
        sb.lineplot(x, z, label="Scipy")
        plt.show()

    def test_beta_visual(self):
        points = 100
        x = np.linspace(0.1, 5, points)
        y = np.linspace(0.1, 5, points)

        X, Y = np.meshgrid(x, y)

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        i_beta = beta(X, Y)
        scipy_beta = beta_(X, Y)

        X = X.reshape(points, points)
        Y = Y.reshape(points, points)

        i_beta = i_beta.reshape(points, points)
        scipy_beta = scipy_beta.reshape(points, points)

        print("Err", np.abs(i_beta - scipy_beta).mean())

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title("Local")
        plt.contourf(X, Y, i_beta)
        plt.subplot(2, 1, 2)
        plt.title("Scipy")
        plt.contourf(X, Y, scipy_beta)
        plt.show()


if __name__ == '__main__':
    unittest.main()
