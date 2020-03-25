import unittest
from probpy.distributions import normal, multivariate_normal
from probpy.inference import posterior
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


class PosteriorTest(unittest.TestCase):
    def test_normal_1d_mean_conjugate(self):
        prior = normal.freeze(mu=1.0, sigma=1.0)
        likelihood = normal.freeze(sigma=2.0)

        data = normal.sample(mu=-2.0, sigma=2.0, shape=10000)
        result = posterior(data, likelihood=likelihood, priors=prior)

        x = np.linspace(-4, 4, 1000)
        y_prior = prior.p(x)
        y_posterior = result.p(x)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior Mean", fontsize=20)
        sb.distplot(data, label="Data")
        sb.lineplot(x, y_prior, label="Prior")
        sb.lineplot(x, y_posterior, label="posterior")
        plt.ylim([0, 1])
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig("../images/normal_1d_mean_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_multivariate_normal_mean_conjugate(self):
        mu_prior = np.ones(2)
        sigma_prior = np.eye(2)

        prior = multivariate_normal.freeze(mu=mu_prior, sigma=sigma_prior)
        likelihood = multivariate_normal.freeze(sigma=np.eye(2) * 10)

        data_mean = np.ones(2) * -2
        data_sigma = np.random.rand(1, 2) * 0.7
        data_sigma = data_sigma.T @ data_sigma + np.eye(2) * 1

        data = multivariate_normal.sample(mu=data_mean, sigma=data_sigma, shape=200)
        result = posterior(data, likelihood=likelihood, priors=prior)

        points = 100
        x = np.linspace(-7, 6, points)
        y = np.linspace(-7, 6, points)

        X, Y = np.meshgrid(x, y)

        mesh = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)

        y_prior = prior.p(mesh).reshape(points, points)
        y_posterior = result.p(mesh).reshape(points, points)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior Mean", fontsize=20)
        plt.contourf(X, Y, y_posterior, levels=10)
        sb.scatterplot(data[:, 0], data[:, 1], label="Data")
        plt.contour(X, Y, y_prior, levels=10)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig("../images/multinormal_mean_conjugate.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
