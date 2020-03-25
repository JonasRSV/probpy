import unittest
from probpy.distributions import (normal,
                                  multivariate_normal,
                                  exponential,
                                  beta,
                                  bernoulli,
                                  categorical,
                                  dirichlet,
                                  gamma)
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

    def test_bernoulli_beta_conjugate(self):
        prior = beta.freeze(a=1.0, b=3.0)
        likelihood = bernoulli.freeze()

        data = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        result = posterior(data, likelihood=likelihood, priors=prior)

        x = np.linspace(0, 1, 100)
        y_prior = prior.p(x)
        y_posterior = result.p(x)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior", fontsize=20)
        sb.lineplot(x, y_prior, label="Prior")
        sb.lineplot(x, y_posterior, label="posterior")
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.savefig("../images/bernoulli_beta_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_categorical_dirichlet_conjugate(self):
        prior = dirichlet.freeze(alpha=np.ones(5))
        likelihood = categorical.freeze(dim=5)

        data = np.array([0, 1, 2, 1, 2, 3, 4, 1])
        result = posterior(data, likelihood=likelihood, priors=prior)

        x = np.arange(5)
        y_prior = prior.sample(shape=10000).sum(axis=0)
        y_posterior = result.sample(shape=10000).sum(axis=0)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior", fontsize=20)
        plt.subplot(1, 2, 1)
        sb.barplot(x, y_prior, label="Prior")
        plt.xticks(fontsize=18)
        plt.legend(fontsize=20)
        plt.subplot(1, 2, 2)
        sb.barplot(x, y_posterior, label="posterior")
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.xticks(fontsize=18)
        plt.savefig("../images/categorical_dirichlet_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_exponential_gamma_conjugate(self):
        prior = gamma.freeze(a=9, b=2)
        likelihood = exponential.freeze()

        data = exponential.sample(lam=1, shape=100)
        result = posterior(data, likelihood=likelihood, priors=prior)

        x = np.linspace(0, 14, 1000)
        y_prior = prior.p(x)
        y_posterior = result.p(x)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior", fontsize=20)
        sb.distplot(data, label="Data")
        sb.lineplot(x, y_prior, label="Prior")
        sb.lineplot(x, y_posterior, label="posterior")
        plt.tight_layout()
        plt.xticks(fontsize=18)
        plt.legend(fontsize=20)
        plt.ylim([0, 2])
        plt.savefig("../images/exponential_gamma_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_should_fail(self):
        prior = exponential.freeze(lam=1.0)
        likelihood = normal.freeze(sigma=2.0)

        data = normal.sample(mu=-2.0, sigma=2.0, shape=10000)
        posterior(data, likelihood=likelihood,
                  priors=prior)  # Should fail because exponential is not conjugate to normal


if __name__ == '__main__':
    unittest.main()
