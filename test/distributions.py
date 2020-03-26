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
from probpy.distributions import gamma
from probpy.distributions import normal_inverse_gamma
from probpy.distributions import geometric
from probpy.distributions import poisson
from probpy.distributions import hypergeometric
from collections import Counter
import itertools


class TestDistributions(unittest.TestCase):

    def test_r(self):
        samples = hypergeometric.sample(N=10, K=5, n=3, shape=5)

        print(samples)

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

        a, b = 9.0, 2.0
        n = gamma.sample(a, b, 10000)
        self.assertAlmostEqual(n.mean(), a / b, delta=1e-1)

        mu, lam, a, b = 2.0, 1.0, 2.0, 2.0
        n = normal_inverse_gamma.sample(mu, lam, a, b, shape=10000)
        self.assertAlmostEqual(n[:, 0].mean(), mu, delta=1e-1)
        self.assertAlmostEqual(n[:, 1].mean(), b / (a - 1), delta=1e-1)

        p = 0.7
        n = geometric.sample(probability=p, shape=10000)
        self.assertAlmostEqual(n.mean(), 1 / p, delta=1e-1)
        self.assertAlmostEqual(n.var(), (1 - p) / (p * p), delta=1e-1)

        lam = 4.0
        n = poisson.sample(lam=lam, shape=100000)
        self.assertAlmostEqual(n.mean(), lam, delta=1e-1)
        self.assertAlmostEqual(n.var(), lam, delta=1e-1)

        N, K, n = 10, 5, 3
        _n = hypergeometric.sample(N=N, K=K, n=n, shape=100000)
        self.assertAlmostEqual(_n.mean(), n * K / N, delta=1e-1)
        self.assertAlmostEqual(_n.var(), n * (K / N) * ((N - K) / N) * (N - n) / (N - 1), delta=1e-1)

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

        x = np.linspace(-1.2, 1.2, 100)
        n = normal.p(x, mu, sigma)
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
        sb.kdeplot(n[:, 0], n[:, 1], shade=True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        points = 100
        x = np.linspace(-1, 3, points)
        y = np.linspace(-4, 4, points)

        X, Y = np.meshgrid(x, y)

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        I = np.concatenate([X, Y], axis=1)
        P = multivariate_normal.p(I, mu, sigma)

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

        x = np.linspace(-3, 4, 100)
        n = uniform.p(x, a, b)
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
        sb.kdeplot(n[:, 0], n[:, 1], shade=True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        points = 1000
        x = np.linspace(-2.5, 2.5, points)
        y = np.linspace(-1.5, 3.5, points)

        X, Y = np.meshgrid(x, y)

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        I = np.concatenate([X, Y], axis=1)
        P = multivariate_uniform.p(I, a, b)

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
        sb.barplot([0, 1], [bernoulli.p(0.0, p), bernoulli.p(1.0, p)])
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
        sb.barplot(np.arange(p.size), [categorical.p(c, p) for c in np.arange(p.size)])
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

        p = dirichlet.p(x, alpha)
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
        x = np.linspace(0.01, 0.6, 100)
        y = beta.p(x, a, b)
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
        y = exponential.p(x, lam)
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
        y = binomial.p(x, n, p)
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

        combinations = np.array(
            [[i, j, k] for i in range(n + 1) for j in range(n + 1) for k in range(n + 1) if i + j + k == n])
        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = combinations
        y = multinomial.p(x, n, p)
        x = (x @ projection).reshape(-1)
        sb.lineplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/multinomial.png", bbox_inches="tight")
        plt.show()

        plt.show()

    def test_gamma_by_inspection(self):
        a, b = 9, 2
        samples = 10000
        n = gamma.sample(a, b, samples)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"a: {a} b: {b}-- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.distplot(n)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = np.linspace(0.0, 14, 100)
        y = gamma.p(x, a, b)
        sb.lineplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/gamma.png", bbox_inches="tight")
        plt.show()

    def test_normal_inverse_gamma_by_inspection(self):
        samples = 10000
        mu, lam, a, b = 2.0, 1.0, 2.0, 2.0
        n = normal_inverse_gamma.sample(mu, lam, a, b, shape=samples)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"mu: {mu} lam: {lam} a: {a} b: {b}-- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.kdeplot(n[:, 0], n[:, 1], shade=True)
        plt.ylim([0, 6])
        plt.xlim([-2, 6])

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)

        points = 100
        x = np.linspace(-2, 6, points)
        y = np.linspace(0.1, 6, points)

        X, Y = np.meshgrid(x, y)
        mesh = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)

        Z = normal_inverse_gamma.p(mesh, mu, lam, a, b).reshape(points, points)
        plt.contourf(X, Y, Z, levels=20)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/normal_inverse_gamma.png", bbox_inches="tight")
        plt.show()

    def test_geometric_by_inspection(self):
        samples = 10000
        p = 0.7
        n = geometric.sample(p, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Geometric: p {p} -- samples: {samples}", fontsize=20)
        sb.distplot(n)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        sb.barplot([1, 2, 3, 4, 5, 6, 7, 8], geometric.p(np.arange(1, 9), p))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/geometric.png", bbox_inches="tight")
        plt.show()

    def test_poisson_by_inspection(self):
        lam = 4.0
        samples = 10000
        n = poisson.sample(lam, samples)

        k, counts = zip(*sorted(list(Counter(n).items()), key=lambda x: x[0]))
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"lambda: {lam} -- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.barplot(list(k), list(counts))

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = np.arange(0, 15)
        y = poisson.p(x, lam)
        sb.barplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/poisson.png", bbox_inches="tight")
        plt.show()

    def test_hypergeometric_by_inspection(self):
        samples = 100000
        N, K, n = 20, 13, 12
        _n = hypergeometric.sample(N=N, K=K, n=n, shape=samples)

        k, counts = zip(*sorted(list(Counter(_n).items()), key=lambda x: x[0]))
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"N: {N} K: {K} n: {n}-- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.barplot(list(k), list(counts))

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        x = np.arange(0, 13)
        y = hypergeometric.p(x, N, K, n)
        sb.barplot(x, y)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/hypergeometric.png", bbox_inches="tight")
        plt.show()

    def test_freezing(self):
        frozen = normal.freeze(mu=0.6, sigma=0.9)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = normal.freeze(mu=0.6)
        s2 = frozen.sample(0.9, shape=100000)
        p2 = frozen.p(s2, 0.9)

        frozen = normal.freeze(sigma=0.9)
        s3 = frozen.sample(0.6, shape=100000)
        p3 = frozen.p(s3, 0.6)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-1)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-1)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-1)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-1)

        frozen = multivariate_normal.freeze(mu=np.zeros(2), sigma=np.eye(2))
        s1 = frozen.sample(shape=10000)
        p1 = frozen.p(s1)

        frozen = multivariate_normal.freeze(mu=np.zeros(2))
        s2 = frozen.sample(np.eye(2), shape=10000)
        p2 = frozen.p(s2, np.eye(2))

        frozen = multivariate_normal.freeze(sigma=np.eye(2))
        s3 = frozen.sample(np.zeros(2), shape=10000)
        p3 = frozen.p(s3, np.zeros(2))

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-1)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-1)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-1)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-1)

        frozen = uniform.freeze(a=0.0, b=1.0)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = uniform.freeze(a=0.0)
        s2 = frozen.sample(1.0, shape=100000)
        p2 = frozen.p(s2, 1.0)

        frozen = uniform.freeze(b=1.0)
        s3 = frozen.sample(0.0, shape=100000)
        p3 = frozen.p(s3, 0.0)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-2)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-2)

        frozen = multivariate_uniform.freeze(a=np.zeros(2), b=np.ones(2))
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = multivariate_uniform.freeze(a=np.zeros(2))
        s2 = frozen.sample(np.ones(2), shape=100000)
        p2 = frozen.p(s2, np.ones(2))

        frozen = multivariate_uniform.freeze(b=np.ones(2))
        s3 = frozen.sample(np.zeros(2), shape=100000)
        p3 = frozen.p(s3, np.zeros(2))

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-2)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-2)

        frozen = bernoulli.freeze(probability=0.5)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = bernoulli.freeze()
        s2 = frozen.sample(0.5, shape=100000)
        p2 = frozen.p(s2, 0.5)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)

        frozen = beta.freeze(a=2.0, b=1.0)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = beta.freeze(a=2.0)
        s2 = frozen.sample(1.0, shape=100000)
        p2 = frozen.p(s2, 1.0)

        frozen = beta.freeze(b=1.0)
        s3 = frozen.sample(2.0, shape=100000)
        p3 = frozen.p(s3, 2.0)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-2)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-2)

        frozen = binomial.freeze(n=3, probability=0.5)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = binomial.freeze(n=3)
        s2 = frozen.sample(0.5, shape=100000)
        p2 = frozen.p(s2, 0.5)

        frozen = binomial.freeze(probability=0.5)
        s3 = frozen.sample(3, shape=100000)
        p3 = frozen.p(s3, 3)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-2)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-2)

        frozen = categorical.freeze(probabilities=np.ones(2) * 0.5)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = categorical.freeze()
        s2 = frozen.sample(np.ones(2) * 0.5, shape=100000)
        p2 = frozen.p(s2, np.ones(2) * 0.5)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)

        frozen = dirichlet.freeze(alpha=np.ones(2) * 2.0)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = dirichlet.freeze()
        s2 = frozen.sample(np.ones(2) * 2.0, shape=100000)
        p2 = frozen.p(s2, np.ones(2) * 2.0)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)

        frozen = exponential.freeze(lam=1.0)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = exponential.freeze()
        s2 = frozen.sample(1.0, shape=100000)
        p2 = frozen.p(s2, 1.0)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)

        p = np.ones(4) * 0.25
        frozen = multinomial.freeze(n=3, probabilities=p)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = multinomial.freeze(n=3)
        s2 = frozen.sample(p, shape=100000)
        p2 = frozen.p(s2, p)

        frozen = multinomial.freeze(probabilities=p)
        s3 = frozen.sample(3, shape=100000)
        p3 = frozen.p(s3, 3)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-2)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-2)

        frozen = gamma.freeze(a=2.0, b=1.0)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = gamma.freeze(a=2.0)
        s2 = frozen.sample(1.0, shape=100000)
        p2 = frozen.p(s2, 1.0)

        frozen = gamma.freeze(b=1.0)
        s3 = frozen.sample(2.0, shape=100000)
        p3 = frozen.p(s3, 2.0)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-1)
        self.assertAlmostEqual(s2.mean(), s3.mean(), delta=1e-1)

        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-1)
        self.assertAlmostEqual(p2.mean(), p3.mean(), delta=1e-1)

        parameters = [2, 1, 2, 2]
        name = [normal_inverse_gamma.mu,
                normal_inverse_gamma.lam,
                normal_inverse_gamma.a,
                normal_inverse_gamma.b]

        s_results = []
        p_results = []
        for test_case in itertools.product([0, 1], repeat=4):
            kwargs = {}
            args = []
            for i, pick in enumerate(test_case):
                if pick == 1:
                    kwargs[name[i]] = parameters[i]
                else:
                    args.append(parameters[i])

            frozen = normal_inverse_gamma.freeze(**kwargs)
            s = frozen.sample(*args, shape=100000)
            p = frozen.p(s, *args)

            s_results.append(s.mean())
            p_results.append(p.mean())

        s_results = np.array(s_results)
        p_results = np.array(p_results)

        np.testing.assert_almost_equal(s_results - s_results[0], np.zeros_like(s_results), decimal=1)
        np.testing.assert_almost_equal(p_results - p_results[0], np.zeros_like(p_results), decimal=1)

        frozen = geometric.freeze(probability=0.5)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = geometric.freeze()
        s2 = frozen.sample(0.5, shape=100000)
        p2 = frozen.p(s2, 0.5)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-2)
        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-2)

        frozen = poisson.freeze(lam=2.0)
        s1 = frozen.sample(shape=100000)
        p1 = frozen.p(s1)

        frozen = poisson.freeze()
        s2 = frozen.sample(2.0, shape=100000)
        p2 = frozen.p(s2, 2.0)

        self.assertAlmostEqual(s1.mean(), s2.mean(), delta=1e-1)
        self.assertAlmostEqual(p1.mean(), p2.mean(), delta=1e-1)

        parameters = [10, 5, 3]
        name = [hypergeometric.N,
                hypergeometric.K,
                hypergeometric.n]

        s_results = []
        p_results = []
        for test_case in itertools.product([0, 1], repeat=3):
            kwargs = {}
            args = []
            for i, pick in enumerate(test_case):
                if pick == 1:
                    kwargs[name[i]] = parameters[i]
                else:
                    args.append(parameters[i])

            frozen = hypergeometric.freeze(**kwargs)
            s = frozen.sample(*args, shape=100000)
            p = frozen.p(s, *args)

            s_results.append(s.mean())
            p_results.append(p.mean())

        s_results = np.array(s_results)
        p_results = np.array(p_results)

        np.testing.assert_almost_equal(s_results - s_results[0], np.zeros_like(s_results), decimal=1)
        np.testing.assert_almost_equal(p_results - p_results[0], np.zeros_like(p_results), decimal=1)


if __name__ == '__main__':
    unittest.main()
