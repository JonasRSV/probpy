import unittest
from probpy.distributions import (normal,
                                  multivariate_normal,
                                  uniform,
                                  multivariate_uniform,
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

    def test_normal_mean_exponential_prior(self):
        prior = uniform.med(a=0.0, b=3.0)

        likelihood = normal.med(sigma=0.5)

        x = np.linspace(-2.0, 4.0, 20)

        probability = predictive_posterior(likelihood=likelihood, priors=prior, data=x)

        plt.figure(figsize=(10, 6))
        sb.lineplot(x, probability)
        plt.show()

    def test_multinormal_mean_multiuniform_prior(self):
        prior = multivariate_uniform.med(a=np.ones(2) * 0, b=np.ones(2) * 2)
        likelihood = multivariate_normal.med(sigma=np.eye(2))

        grid = 20
        i = np.linspace(-2.0, 4.0, grid)
        j = np.linspace(-2.0, 4.0, grid)

        I, J = np.meshgrid(i, j)
        X = np.concatenate([I.reshape(-1, 1), J.reshape(-1, 1)], axis=1)

        probability = predictive_posterior(likelihood=likelihood, priors=prior, data=X)

        plt.figure(figsize=(10, 6))
        plt.contourf(I, J, probability.reshape(grid, grid))
        plt.show()

    def test_unilinear_multinormal_prior(self):
        prior = multivariate_normal.med(mu=np.array([1.0, 0.5]), sigma=np.eye(2) * 1e-1)
        likelihood = unilinear.med(sigma=0.3)

        x = np.array(0.8)
        y = np.linspace(-2, 4, 20)

        probability = predictive_posterior(likelihood=likelihood, priors=prior, data=(y, x))

        print(probability)
        plt.figure(figsize=(10, 6))
        sb.lineplot(y, probability)
        plt.xlabel("Y", fontsize=15)
        plt.ylabel("Probability", fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig("../images/unilinear-multinormal-predictive-posterior.png")
        plt.show()

    def test_custom_likelihood_function(self):
        prior = uniform.med(a=-4, b=4)

        def likelihood(y, x, w):
            result = []
            for _w in w:
                result.append(
                    normal.p(_w - np.float_power(y, x), mu=0.0, sigma=0.001)
                )

            return np.array(result)

        x = np.array(2.0)
        y = np.linspace(-3, 3, 20)

        probability = predictive_posterior(likelihood=likelihood, priors=prior, data=(y, x), size=10000)

        plt.figure(figsize=(10, 6))
        sb.lineplot(y, probability)
        plt.xlabel("Y", fontsize=15)
        plt.ylabel("Probability", fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig("../images/custom-odd-predictive-posterior.png")
        plt.show()

    def test_custom_likelihood_function_logistic_regression(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def likelihood(y, x, w):
            return normal.p((y - sigmoid(x @ w[:, None, :-1] + w[:, None, None, -1]).squeeze(axis=2)),
                            mu=0.0, sigma=0.01)

        x = np.array(0.6).reshape(1, 1)
        y = np.linspace(-1, 3, 50)

        probability = predictive_posterior(likelihood=likelihood,
                                           priors=multivariate_normal.med(mu=np.array([1.5, 0.1]),
                                                                          sigma=np.eye(2)),
                                           data=(y, x),
                                           size=10000)

        plt.figure(figsize=(10, 6))
        sb.lineplot(y, probability)
        plt.xlabel("Y", fontsize=15)
        plt.ylabel("Probability", fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig("../images/custom-logistic-regression-predictive-posterior.png")
        plt.show()


if __name__ == '__main__':
    unittest.main()
