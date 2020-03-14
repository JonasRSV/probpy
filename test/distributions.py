import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from probpy.distributions import normal, multivariate_normal
from probpy.distributions import uniform, multivariate_uniform
from probpy.distributions import bernoulli
from probpy.distributions import categorical
from probpy.distributions import dirichlet
from probpy.distributions import beta
from probpy.distributions import exponential
from probpy.distributions import binomial
from probpy.distributions import multinomial


class TestDistributions(unittest.TestCase):

    def test_freezing(self):
        frozen = normal.freeze(mu=0.6, sigma=0.9)

        plt.figure(figsize=(10, 6))
        sb.distplot(frozen.sample(shape=100))
        plt.show()

    def test_first_two_moments(self):
        # Error should be variance / sqrt(n)
        mu, sigma = 0, 2.0
        n = normal.sample(mu, sigma, 30000)
        self.assertAlmostEqual(n.mean(), mu, delta=1e-1)
        self.assertAlmostEqual(n.var(), sigma, delta=1e-1)

        mu = np.array([0.0, 0.0])
        sigma = np.eye(2)
        n = multivariate_normal.sample(mu, sigma, 20000)
        np.testing.assert_almost_equal(n.mean(axis=0), mu, decimal=1)
        np.testing.assert_almost_equal(np.cov(n, rowvar=False), sigma, decimal=1)

        a, b = 0, 1
        n = uniform.sample(a, b, 10000)
        self.assertAlmostEqual(n.mean(), (b - a) / 2, delta=1e-1)
        self.assertAlmostEqual(n.var(), (1 / 12) * np.square(b - a), delta=1e-1)

        a = np.array([0, 0])
        b = np.array([1, 1])
        n = multivariate_uniform.sample(a, b, 10000)
        np.testing.assert_almost_equal(n.mean(axis=0), (b - a) / 2, decimal=1)
        np.testing.assert_almost_equal(np.cov(n, rowvar=False).sum(axis=1), (1 / 12) * np.square(b - a), decimal=1)

        p = 0.5
        n = bernoulli.sample(p, 10000)
        self.assertAlmostEqual(n.mean(), p, delta=1e-1)
        self.assertAlmostEqual(n.var(), p * (1 - p), delta=1e-1)

        p = np.array([0.3, 0.5, 0.2])
        n = categorical.one_hot(categorical.sample(p, 30000), 3)
        np.testing.assert_almost_equal(n.mean(axis=0), p, decimal=1)
        np.testing.assert_almost_equal(np.diag(np.cov(n, rowvar=False)), p * (1 - p), decimal=1)

        alphas = np.array([1.0, 2.0, 1.0])
        n = dirichlet.sample(alphas, 10000)
        alpha_i = alphas / alphas.sum()
        np.testing.assert_almost_equal(n.mean(axis=0), alpha_i, decimal=1)
        np.testing.assert_almost_equal(np.diag(np.cov(n, rowvar=False)), alpha_i * (1 - alpha_i) / (alphas.sum() + 1),
                                       decimal=1)

        a, b = 1, 1
        n = beta.sample(a, b, 10000)
        self.assertAlmostEqual(n.mean(), a / (a + b), delta=1e-1)
        self.assertAlmostEqual(n.var(), (a * b) / (np.square(a + b) * (a + b + 1)), delta=1e-1)

        lam = 2
        n = exponential.sample(lam, 10000)
        self.assertAlmostEqual(n.mean(), 1 / lam, delta=1e-1)
        self.assertAlmostEqual(n.var(), 1 / np.square(lam), delta=1e-1)

        # Testing only first moment here
        p, n = 0.7, 20
        _n = binomial.sample(n, p, 10000)
        self.assertAlmostEqual((_n / n).mean(), p, delta=1e-1)

        p, n = np.array([0.3, 0.4, 0.3]), 20
        _n = multinomial.sample(n, p, 10000)
        np.testing.assert_almost_equal((_n / n).mean(axis=0), p, decimal=1)

    def test_normal_by_inspection(self):
        samples = 10000
        mu, sigma = 0, 0.1
        n = normal.sample(mu, sigma, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Normal: mu {mu} - sigma {sigma} -- samples {samples}", fontsize=20)
        sb.distplot(n)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        x = np.linspace(-4, 4, 100)
        n = normal.p(mu, sigma, x)
        plt.subplot(2, 1, 2)
        plt.title("True", fontsize=20)
        sb.lineplot(x, n)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/normal.png", bbox_inches="tight")
        plt.show()

    def test_multi_normal_by_inspection(self):
        samples = 10000
        mu = np.zeros(2)
        mu[0] = 1
        sigma = np.eye(2)
        sigma[0, 0] = 0.3
        sigma[0, 1] = 0.5
        sigma[1, 0] = 0.5

        n = multivariate_normal.sample(mu, sigma, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Multivariate normal: mu {mu} - sigma {sigma} -- samples: {samples}", fontsize=20)
        sb.kdeplot(n[:, 0], n[:, 1])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

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
        plt.title(f"True", fontsize=20)
        plt.contourf(X, Y, P)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/multi_normal.png", bbox_inches="tight")
        plt.show()

    def test_uniform_by_inspection(self):
        samples = 10000
        a, b = -2, 3
        n = uniform.sample(a, b, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"uniform: a {a} - b {b} -- samples: {samples}", fontsize=20)
        sb.distplot(n)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        x = np.linspace(-4, 4, 100)
        n = uniform.p(a, b, x)
        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        sb.lineplot(x, n)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/uniform.png", bbox_inches="tight")
        plt.show()

    def test_multivariate_uniform_by_inspection(self):
        samples = 10000
        a = np.array([-2, -1])
        b = np.array([2, 3])

        n = multivariate_uniform.sample(a, b, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Multivariate uniform: a {a} - b {b} -- samples: {samples}", fontsize=20)
        sb.kdeplot(n[:, 0], n[:, 1])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

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
        plt.title(f"True", fontsize=20)
        plt.contourf(X, Y, P)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/multi_uniform.png", bbox_inches="tight")
        plt.show()

    def test_bernoulli_by_inspection(self):
        samples = 10000
        p = 0.7
        n = bernoulli.sample(p, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Bernoulli: p {p} -- samples: {samples}", fontsize=20)
        sb.barplot([0, 1], [sum(n == 0.0) / len(n), sum(n == 1.0) / len(n)])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        sb.barplot([0, 1], [bernoulli.p(p, 0.0), bernoulli.p(p, 1.0)])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/bernoulli.png", bbox_inches="tight")
        plt.show()

    def test_categorical_by_inspection(self):
        samples = 10000
        p = np.array([0.3, 0.6, 0.1])
        n = categorical.sample(p, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Categorical: {p[0]} - {p[1]} - {p[2]} -- samples: {samples}", fontsize=20)
        sb.barplot(np.arange(p.size), [sum(n == c) / len(n) for c in np.arange(p.size)])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        sb.barplot(np.arange(p.size), [categorical.p(p, c) for c in np.arange(p.size)])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/categorical.png", bbox_inches="tight")
        plt.show()

    def test_dirichlet_by_inspection(self):
        samples = 10000
        alpha = np.array([2.0, 3.0])
        n = dirichlet.sample(alpha, samples)

        projection = np.ones(2)
        projection[0] = -1
        projection[1] = 1
        n = n @ projection

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"projection of alphas: {alpha[0]}, {alpha[1]} -- samples: {samples}", fontsize=20)
        sb.distplot(n)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        x = np.linspace(0, 1, 100).reshape(-1, 1)
        x = np.concatenate([x, x[::-1]], axis=1)

        p = dirichlet.p(alpha, x)
        x = x @ projection
        plt.subplot(2, 1, 2)
        plt.title(f"True projection", fontsize=20)
        sb.lineplot(x, p)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/dirichlet.png", bbox_inches="tight")
        plt.show()

    def test_beta_by_inspection(self):
        samples = 10000
        a, b = np.array(10.0), np.array(30.0)
        n = beta.sample(a, b, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"alpha: {a} -- beta: {b} -- samples: {samples}", fontsize=20)
        sb.distplot(n)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = np.linspace(0.01, 0.99, 100)
        y = beta.p(a, b, x)
        sb.lineplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/beta.png", bbox_inches="tight")
        plt.show()

    def test_exponential_by_inspection(self):
        lam = 2
        samples = 10000
        n = exponential.sample(lam, samples)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"lambda: {lam} -- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.distplot(n)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = np.linspace(0.0, 5, 100)
        y = exponential.p(lam, x)
        sb.lineplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/exponential.png", bbox_inches="tight")
        plt.show()

    def test_binomial_by_inspection(self):
        n, p = 20, 0.2
        samples = 10000
        _n = binomial.sample(n, p, samples)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"n: {n} - p: {p} -- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.distplot(_n)
        plt.xlim([0, 20])

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = np.arange(0, 20, dtype=np.int)
        y = binomial.p(n, p, x)
        sb.lineplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/binomial.png", bbox_inches="tight")
        plt.show()

    def test_multinomial_by_inspection(self):
        n, p = 20, np.array([0.3, 0.5, 0.2])
        samples = 10000

        projection = np.random.rand(3, 1) - 0.5
        _n = multinomial.sample(n, p, samples)
        _n = _n @ projection
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"random projection on 1d - n: {n} - p: {p} -- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.distplot(_n)

        combinations = np.array([[i, j, k] for i in range(n + 1) for j in range(n + 1) for k in range(n + 1) if i + j + k == n])
        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = combinations
        y = multinomial.p(n, p, x)
        x = (x @ projection).reshape(-1)
        sb.lineplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/multinomial.png", bbox_inches="tight")
        plt.show()

        plt.show()


if __name__ == '__main__':
    unittest.main()
