import unittest
from probpy.distributions import (normal,
                                  multivariate_normal,
                                  exponential,
                                  beta,
                                  bernoulli,
                                  categorical,
                                  dirichlet,
                                  gamma,
                                  normal_inverse_gamma,
                                  binomial,
                                  multinomial,
                                  poisson,
                                  geometric,
                                  gaussian_process,
                                  unilinear)
from probpy.inference import predictive_posterior
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


class PredictivePosteriorTest(unittest.TestCase):
    def test_bernoulli_beta(self):
        prior = beta.med(a=8.0, b=3.0)
        likelihood = bernoulli.med()

        result = predictive_posterior(likelihood=likelihood, priors=prior)

        x = np.linspace(0, 1, 100)
        y_prior = prior.p(x)
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.title("Parameter Prior", fontsize=20)
        sb.lineplot(x, y_prior, label="Prior")
        plt.legend(fontsize=20)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.subplot(2, 1, 2)
        plt.title("Predictive Posterior", fontsize=20)
        sb.barplot([0, 1], [1 - result.probability, result.probability])
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.tight_layout()
        plt.savefig("../images/bernoulli_beta_predictive.png", bbox_inches="tight")
        plt.show()

    def test_normal_normal(self):
        prior = normal.med(mu=1.0, sigma=1.0)
        likelihood = normal.med(sigma=1.0)

        posterior = predictive_posterior(likelihood=likelihood, priors=prior)

        prior_samples = prior.sample(size=100000)
        posterior_samples = posterior.sample(size=100000)

        plt.figure(figsize=(20, 10))
        sb.distplot(prior_samples, label="prior")
        sb.distplot(posterior_samples, label="posterior")
        plt.legend(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.tight_layout()
        plt.savefig("../images/normal_normal_predictive.png", bbox_inches="tight")
        plt.show()

    def test_multinormal_multinormal(self):

        prior_sigma = np.random.rand(2, 1)
        prior_sigma = prior_sigma.T @ prior_sigma + np.eye(2) * 1

        likelihood_sigma = np.random.rand(2, 1)
        likelihood_sigma = likelihood_sigma.T @ likelihood_sigma + np.eye(2) * 5

        prior = multivariate_normal.med(mu=np.zeros(2), sigma=prior_sigma)
        likelihood = multivariate_normal.med(sigma=likelihood_sigma)

        posterior = predictive_posterior(likelihood=likelihood, priors=prior)

        points = 100
        x = np.linspace(-7, 6, points)
        y = np.linspace(-7, 6, points)

        X, Y = np.meshgrid(x, y)

        mesh = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)

        z_prior = prior.p(mesh).reshape(points, points)
        z_posterior = posterior.p(mesh).reshape(points, points)

        plt.figure(figsize=(20, 10))
        plt.contourf(X, Y, z_prior)
        plt.contour(X, Y, z_posterior)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.tight_layout()
        plt.savefig("../images/multinormal_multinormal_predictive.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
