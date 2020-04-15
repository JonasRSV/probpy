import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time
from probpy.distributions import *
from collections import Counter


class VisualTestDistributions(unittest.TestCase):

    def test_normal_visual(self):
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
        plt.savefig("../../images/normal.png", bbox_inches="tight")
        plt.show()

    def test_multivariate_normal_visual(self):
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
        plt.savefig("../../images/multi_normal.png", bbox_inches="tight")
        plt.show()

    def test_uniform_visual(self):
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
        plt.savefig("../../images/uniform.png", bbox_inches="tight")
        plt.show()

    def test_multivariate_uniform_visual(self):
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

        print("P", P)

        P = P.reshape(points, points)
        X = X.reshape(points, points)
        Y = Y.reshape(points, points)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        plt.contourf(X, Y, P)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../../images/multi_uniform.png", bbox_inches="tight")
        plt.show()

    def test_bernoulli_visual(self):
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
        plt.savefig("../../images/bernoulli.png", bbox_inches="tight")
        plt.show()

    def test_categorical_visual(self):
        samples = 10000
        p = np.array([0.3, 0.6, 0.1])
        n = categorical.sample(p, samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Categorical: {p[0]} - {p[1]} - {p[2]} -- samples: {samples}", fontsize=20)
        sb.barplot(np.arange(p.size), n.sum(axis=0))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(2, 1, 2)
        plt.title(f"True", fontsize=20)
        sb.barplot(np.arange(p.size), categorical.p(np.eye(p.size), p))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../../images/categorical.png", bbox_inches="tight")
        plt.show()

    def test_dirichlet_visual(self):
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
        plt.savefig("../../images/dirichlet.png", bbox_inches="tight")
        plt.show()

    def test_beta_visual(self):
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
        plt.savefig("../../images/beta.png", bbox_inches="tight")
        plt.show()

    def test_exponential_visual(self):
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
        plt.savefig("../../images/exponential.png", bbox_inches="tight")
        plt.show()

    def test_binomial_visual(self):
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
        plt.savefig("../../images/binomial.png", bbox_inches="tight")
        plt.show()

    def test_multinomial_visual(self):
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
        plt.savefig("../../images/multinomial.png", bbox_inches="tight")
        plt.show()

        plt.show()

    def test_gamma_visual(self):
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
        plt.savefig("../../images/gamma.png", bbox_inches="tight")
        plt.show()

    def test_normal_inverse_gamma_visual(self):
        samples = 10000
        mu, lam, a, b = 2.0, 1.0, 2.0, 2.0
        n = normal_inverse_gamma.sample(mu, lam, a, b, size=samples)
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
        plt.savefig("../../images/normal_inverse_gamma.png", bbox_inches="tight")
        plt.show()

    def test_geometric_visual(self):
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
        plt.savefig("../../images/geometric.png", bbox_inches="tight")
        plt.show()

    def test_poisson_visual(self):
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
        plt.savefig("../../images/poisson.png", bbox_inches="tight")
        plt.show()

    def test_hypergeometric_visual(self):
        samples = 100000
        N, K, n = 20, 13, 12
        _n = hypergeometric.sample(N=N, K=K, n=n, size=samples)

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
        plt.savefig("../../images/hypergeometric.png", bbox_inches="tight")
        plt.show()

    def test_gaussian_process_visual(self):

        def mu(x): return 0

        def sigma(x, y): return np.exp(-1.0 * np.square(x - y))

        X = np.array([0.0, 2.0])
        Y = np.random.rand(2) * 2
        x = np.linspace(-5, 5, 50)
        samples = 10

        timestamp = time.time()
        n = gaussian_process.sample(x=x, mu=mu, sigma=sigma, X=X, Y=Y, size=samples)
        print("Time to sample", time.time() - timestamp)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"RBF Gaussian Processes", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        for _n in n:
            sb.lineplot(x, _n)

        plt.subplot(2, 1, 2)
        plt.title("Probabilities of functions values above at x = 1.0", fontsize=20)
        probabilities = gaussian_process.p(np.linspace(-2.0, 3.0, 1000), x=np.array([1.0]), mu=mu, sigma=sigma, X=X,
                                           Y=Y)
        sb.lineplot(np.linspace(-2, 3, 1000), probabilities)
        plt.xticks(fontsize=20)
        plt.xlabel("Y", fontsize=20)
        plt.tight_layout()
        plt.savefig("../../images/gaussian_process.png", bbox_inches="tight")
        plt.show()

    def test_unilinear_visual(self):

        x = np.linspace(0, 2, 100)
        variables = np.ones(2)
        sigma = 0.01
        samples = 100
        y = unilinear.sample(x=x, variables=variables, sigma=sigma, size=samples)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"variables: {variables} sigma: {sigma} -- samples: {samples}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.lineplot(x, y)

        plt.subplot(2, 1, 2)
        plt.title("Densities of different X", fontsize=20)
        p = unilinear.p(y, x=x, variables=variables, sigma=sigma)
        sb.lineplot(x, p)
        plt.tight_layout()
        plt.savefig("../../images/unilinear.png", bbox_inches="tight")
        plt.show()

    def test_function_1d_visual(self):

        def f(x): return np.exp(-np.square(x - 1))

        rv = function.med(density=f, lower_bound=-5, upper_bound=5, points=10000)

        s = 10000
        samples = rv.sample(size=s)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"samples: {s}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.distplot(samples)

        plt.subplot(2, 1, 2)
        plt.title("Densities", fontsize=20)
        x = np.linspace(-1.2, 3, 500)
        y = rv.p(x)
        sb.lineplot(x, y)
        plt.tight_layout()
        plt.savefig("../../images/function_distribution.png", bbox_inches="tight")
        plt.show()

    def test_function_2d_visual(self):
        mean = np.ones(2)

        def f(x): return np.exp(-((x - mean) * (x - mean)).sum(axis=1))

        rv = function.med(density=f, lower_bound=np.ones(2) * -2, upper_bound=np.ones(2) * 2, points=10000)

        s = 10000
        samples = rv.sample(size=s)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"samples: {s}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.kdeplot(samples[:, 0], samples[:, 1], shade=True)

        plt.subplot(2, 1, 2)
        plt.title("Densities", fontsize=20)
        points = 100
        x = np.linspace(-2, 4, points)
        y = np.linspace(-2, 4, points)

        X, Y = np.meshgrid(x, y)

        Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = rv.p(Z)

        plt.contourf(X, Y, Z.reshape(points, points))
        plt.tight_layout()
        plt.savefig("../../images/function_distribution_2d.png", bbox_inches="tight")
        plt.show()

    def test_function_2d_gmm_visual(self):

        m1, m2, m3 = np.ones(2) * 0, np.ones(2) * -3, np.ones(2) * 3

        def f(x):
            return np.exp(-((x - m1) * (x - m1)).sum(axis=1)) \
                   + np.exp(-((x - m2) * (x - m2)).sum(axis=1)) \
                   + np.exp(-((x - m3) * (x - m3)).sum(axis=1))

        lower_bound = np.ones(2) * -4
        upper_bound = np.ones(2) * 4
        rv = function.med(density=f,
                          lower_bound=lower_bound,
                          upper_bound=upper_bound,
                          points=40000,
                          error=1e-1,
                          variance=5.0,
                          verbose=True)

        s = 10000
        samples = rv.sample(size=s)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"samples: {s}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.kdeplot(samples[:, 0], samples[:, 1], shade=True)

        plt.subplot(2, 1, 2)
        plt.title("Densities", fontsize=20)
        points = 100
        x = np.linspace(-6, 6, points)
        y = np.linspace(-6, 6, points)

        X, Y = np.meshgrid(x, y)

        Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = rv.p(Z)

        plt.contourf(X, Y, Z.reshape(points, points))
        plt.tight_layout()
        plt.savefig("../../images/function_distribution_gmm_2d.png", bbox_inches="tight")
        plt.show()

    def test_function_2d_points_visual(self):

        samples = np.concatenate([
            multivariate_normal.sample(mu=np.ones(2) * -2, sigma=np.eye(2), size=10000),
            multivariate_normal.sample(mu=np.ones(2) * 2, sigma=np.eye(2), size=10000),
        ])

        rv = points.med(points=samples, verbose=True)

        s = 10000
        samples = rv.sample(size=s)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"samples: {s}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.kdeplot(samples[:, 0], samples[:, 1], shade=True)

        plt.subplot(2, 1, 2)
        plt.title("Densities", fontsize=20)
        grid = 100
        x = np.linspace(-6, 6, grid)
        y = np.linspace(-6, 6, grid)

        X, Y = np.meshgrid(x, y)

        Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = rv.p(Z)

        plt.contourf(X, Y, Z.reshape(grid, grid))
        plt.tight_layout()
        plt.savefig("../../images/points_distribution_gmm_2d.png", bbox_inches="tight")
        plt.show()



if __name__ == '__main__':
    unittest.main()
