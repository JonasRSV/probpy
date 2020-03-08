import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from probpy.distributions import normal, multivariate_normal
from probpy.distributions import uniform, multivariate_uniform
from probpy.distributions import bernoulli
from probpy.distributions import categorical
from probpy.distributions import dirichlet


class TestDistributions(unittest.TestCase):
    def test_normal(self):
        mu, sigma = 0, 0.1
        n = normal.sample(mu, sigma, 2000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.distplot(n)

        x = np.linspace(-4, 4, 100)
        n = normal.p(mu, sigma, x)
        plt.subplot(2, 1, 2)
        sb.lineplot(x, n)
        plt.show()

    def test_multi_normal(self):
        mu = np.zeros(2)
        mu[0] = 1
        sigma = np.eye(2)
        sigma[0, 0] = 0.3
        sigma[0, 1] = 0.5
        sigma[1, 0] = 0.5

        n = multivariate_normal.sample(mu, sigma, 10000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.kdeplot(n[:, 0], n[:, 1])

        points = 100
        x = np.linspace(-4, 4, points)
        y = np.linspace(-4, 4, points)

        X, Y = np.meshgrid(x, y)

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        I = np.concatenate([X, Y], axis=1)
        P = multivariate_normal.p(mu, sigma, I)

        P = P.reshape(points, points)
        X = X.reshape(points, points)
        Y = Y.reshape(points, points)

        plt.subplot(2, 1, 2)
        plt.contourf(X, Y, P)
        plt.show()

    def test_uniform(self):
        a, b = -2, 3
        n = uniform.sample(a, b, 2000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.distplot(n)

        x = np.linspace(-4, 4, 100)
        n = uniform.p(a, b, x)
        plt.subplot(2, 1, 2)
        sb.lineplot(x, n)
        plt.show()

    def test_multivariate_uniform(self):
        a = np.array([-2, -1])
        b = np.array([2, 3])

        n = multivariate_uniform.sample(a, b, 2000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.kdeplot(n[:, 0], n[:, 1])

        points = 1000
        x = np.linspace(-4, 4, points)
        y = np.linspace(-4, 4, points)

        X, Y = np.meshgrid(x, y)

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        I = np.concatenate([X, Y], axis=1)
        P = multivariate_uniform.p(a, b, I)

        P = P.reshape(points, points)
        X = X.reshape(points, points)
        Y = Y.reshape(points, points)

        plt.subplot(2, 1, 2)
        plt.contourf(X, Y, P)
        plt.show()

    def test_bernoulli(self):
        p = 0.7
        n = bernoulli.sample(p, 2000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.barplot([0, 1], [sum(n == 0.0) / len(n), sum(n == 1.0) / len(n)])

        plt.subplot(2, 1, 2)
        sb.barplot([0, 1], [bernoulli.p(p, 0.0), bernoulli.p(p, 1.0)])
        plt.show()

    def test_categorical(self):
        p = np.array([0.3, 0.6, 0.1])
        n = categorical.sample(p, 2000)

        print(n[0:3])
        print(categorical.one_hot(n, size=3)[0:3])

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.barplot(np.arange(p.size), [sum(n == c) / len(n) for c in np.arange(p.size)])

        plt.subplot(2, 1, 2)
        sb.barplot(np.arange(p.size), [categorical.p(p, c) for c in np.arange(p.size)])
        plt.show()

    def test_dirichlet(self):
        alpha = np.array([2.0, 3.0])
        n = dirichlet.sample(alpha, 1000)

        projection = np.ones(2)
        projection[0] = -1
        projection[1] = 1
        n = n @ projection

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.distplot(n)

        x = np.linspace(0, 1, 100).reshape(-1, 1)
        x = np.concatenate([x, x[::-1]], axis=1)

        p = dirichlet.p(alpha, x)
        x = x @ projection
        plt.subplot(2, 1, 2)
        sb.lineplot(x, p)
        plt.show()





if __name__ == '__main__':
    unittest.main()
