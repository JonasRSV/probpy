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
                                  unilinear,
                                  multivariate_uniform)
from probpy.learn import parameter_posterior
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


class VisualPosteriorTest(unittest.TestCase):
    def test_normal_1d_mean_conjugate_visual(self):
        prior = normal.med(mu=1.0, sigma=1.0)
        likelihood = normal.med(sigma=2.0)

        data = normal.sample(mu=-2.0, sigma=2.0, size=10000)
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

        x = np.linspace(-4, 4, 1000)
        y_prior = prior.p(x)
        y_posterior = result.p(x)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior Mean", fontsize=20)
        sb.lineplot(x, y_prior, label="Prior")
        sb.lineplot(x, y_posterior, label="posterior")
        plt.ylim([0, 1])
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig("../../images/normal_1d_mean_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_normal_1d_normal_inverse_gamma_conjugate_visual(self):
        prior = normal_inverse_gamma.med(mu=1.0, lam=2.0, a=3.0, b=3.0)
        likelihood = normal.med()

        data = normal.sample(mu=-2.0, sigma=2.0, size=100)
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

        points = 100
        x = np.linspace(-3, 3, points)
        y = np.linspace(0.1, 6, points)

        X, Y = np.meshgrid(x, y)
        mesh = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)

        z_prior = prior.p(mesh).reshape(points, points)
        z_posterior = result.p(mesh).reshape(points, points)

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.title("Prior and Posterior", fontsize=20)
        plt.contourf(X, Y, z_posterior)
        plt.contour(X, Y, z_prior)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.subplot(2, 1, 2)
        plt.title("Data", fontsize=20)
        sb.distplot(data)
        plt.xticks(fontsize=25)
        plt.tight_layout()
        plt.savefig("../../images/normal_1d_normal_inverse_gamma_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_multivariate_normal_mean_conjugate_visual(self):
        mu_prior = np.ones(2)
        sigma_prior = np.eye(2)

        prior = multivariate_normal.med(mu=mu_prior, sigma=sigma_prior)
        likelihood = multivariate_normal.med(sigma=np.eye(2) * 10)

        data_mean = np.ones(2) * -2
        data_sigma = np.random.rand(1, 2) * 0.7
        data_sigma = data_sigma.T @ data_sigma + np.eye(2) * 1

        data = multivariate_normal.sample(mu=data_mean, sigma=data_sigma, size=200)
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

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
        plt.savefig("../../images/multinormal_mean_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_bernoulli_beta_conjugate_visual(self):
        prior = beta.med(a=1.0, b=3.0)
        likelihood = bernoulli.med()

        data = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

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
        plt.savefig("../../images/bernoulli_beta_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_categorical_dirichlet_conjugate_visual(self):
        prior = dirichlet.med(alpha=np.ones(5))
        likelihood = categorical.med(dim=5)

        data = np.array([0, 1, 2, 1, 2, 3, 4, 1])
        data = np.eye(5)[data]
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

        x = np.arange(5)
        y_prior = prior.sample(size=10000).sum(axis=0)
        y_posterior = result.sample(size=10000).sum(axis=0)

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
        plt.savefig("../../images/categorical_dirichlet_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_exponential_gamma_conjugate_visual(self):
        prior = gamma.med(a=9, b=2)
        likelihood = exponential.med()

        data = exponential.sample(lam=1, size=100)
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

        x = np.linspace(0, 14, 1000)
        y_prior = prior.p(x)
        y_posterior = result.p(x)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior", fontsize=20)
        sb.lineplot(x, y_prior, label="Prior")
        sb.lineplot(x, y_posterior, label="posterior")
        plt.tight_layout()
        plt.xticks(fontsize=18)
        plt.legend(fontsize=20)
        plt.ylim([0, 2])
        plt.savefig("../../images/exponential_gamma_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_binomial_beta_conjugate_visual(self):
        prior = beta.med(a=6.0, b=3.0)
        likelihood = binomial.med(n=5)

        data = np.array([0, 2, 4, 1, 1, 0])
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

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
        plt.savefig("../../images/binomial_beta_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_multinomial_dirichlet_conjugate_visual(self):
        prior = dirichlet.med(alpha=np.ones(3))
        likelihood = multinomial.med(n=3)

        data = np.array([[1, 1, 1], [0, 2, 1], [0, 0, 3], [0, 0, 3]])
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

        x = np.arange(3)
        y_prior = prior.sample(size=10000).sum(axis=0)
        y_posterior = result.sample(size=10000).sum(axis=0)

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
        plt.savefig("../../images/multinomial_dirichlet_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_poisson_gamma_conjugate_visual(self):
        prior = gamma.med(a=9, b=2)
        likelihood = poisson.med()

        data = poisson.sample(lam=2.0, size=40)
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

        x = np.linspace(0, 14, 1000)
        y_prior = prior.p(x)
        y_posterior = result.p(x)

        plt.figure(figsize=(20, 10))
        plt.title("Estimating Posterior", fontsize=20)
        sb.lineplot(x, y_prior, label="Prior")
        sb.lineplot(x, y_posterior, label="posterior")
        plt.tight_layout()
        plt.xticks(fontsize=18)
        plt.legend(fontsize=20)
        plt.ylim([0, 2])
        plt.savefig("../../images/poisson_gamma_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_geometric_beta_conjugate_visual(self):
        prior = beta.med(a=6.0, b=3.0)
        likelihood = geometric.med()

        data = np.array([0, 2, 4, 1, 1, 0, 8, 8, 9, 8, 10, 10, 10, 10])
        result = parameter_posterior(data, likelihood=likelihood, priors=prior)

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
        plt.savefig("../../images/geometric_beta_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_unilinear_multivariate_normal_conjugate_visual(self):
        prior = multivariate_normal.med(mu=np.ones(2) * -1, sigma=np.eye(2) * 1e-1)
        likelihood = unilinear.med(sigma=1e-1)

        variables = np.array([2, 1])
        x = np.linspace(-1, 1, 300)
        y = unilinear.sample(x=x, variables=variables, sigma=1e-1)
        posterior = parameter_posterior((y, x), likelihood=likelihood, priors=prior)

        prior_samples = prior.sample(size=10000)
        posterior_samples = posterior.sample(size=10000)

        prior_mean = prior_samples.mean(axis=0)
        posterior_mean = posterior_samples.mean(axis=0)

        prior_y = unilinear.sample(x=x, variables=prior_mean, sigma=1e-1)
        posterior_y = unilinear.sample(x=x, variables=posterior_mean, sigma=1e-1)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"Samples", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        sb.lineplot(x, y, label="data")
        sb.lineplot(x, prior_y, label="Prior")
        sb.lineplot(x, posterior_y, label="Posterior")
        plt.legend(fontsize=18)

        plt.subplot(2, 1, 2)
        points = 100
        x = np.linspace(-7, 6, points)
        y = np.linspace(-7, 6, points)

        X, Y = np.meshgrid(x, y)

        mesh = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)

        y_prior = prior.p(mesh).reshape(points, points)
        y_posterior = posterior.p(mesh).reshape(points, points)

        plt.title("Parameter distributions", fontsize=20)
        plt.contourf(X, Y, y_posterior, levels=10)
        plt.contour(X, Y, y_prior, levels=10)
        plt.tight_layout()
        plt.xlim([-2, 3])
        plt.ylim([-2, 3])
        plt.savefig("../../images/unilinear_multivariate_gaussian_conjugate.png", bbox_inches="tight")
        plt.show()

    def test_normal_with_exponential_prior_mcmc_visual(self):
        prior = exponential.med(lam=1.0)
        likelihood = normal.med(sigma=2.0)

        data = normal.sample(mu=3.0, sigma=2.0, size=100)
        posterior = parameter_posterior(data, likelihood=likelihood, priors=prior, size=2000)

        x = np.linspace(0.0, 6, 100)
        y_prior = prior.p(x)
        y_posterior = posterior.p(x)

        plt.figure(figsize=(20, 10))
        sb.lineplot(x, y_prior, label="prior")
        sb.lineplot(x, y_posterior, label="posterior")
        plt.legend()
        plt.show()

    def test_normal_with_exponential_normal_prior_mcmc(self):
        mu_prior = normal.med(mu=3.0, sigma=2.0)
        exp_prior = exponential.med(lam=1.0)
        likelihood = normal.med()

        data = normal.sample(mu=5.0, sigma=2.0, size=300)
        posterior = parameter_posterior(data, likelihood=likelihood, priors=(mu_prior, exp_prior), size=2000)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        sb.distplot(data, label="data")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title("Posterior", fontsize=20)
        grid = 100
        x = np.linspace(3, 7.0, grid)
        y = np.linspace(-2.0, 6.0, grid)

        X, Y = np.meshgrid(x, y)
        Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = posterior.p(Z)

        plt.contourf(X, Y, Z.reshape(grid, grid))
        plt.tight_layout()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig("../../images/normal-exponential-normal-mcmc-prior.png", bbox_inches="tight")
        plt.show()

    def test_multinormal_with_multiuniform_prior_mcmc(self):
        prior = multivariate_uniform.med(a=np.zeros(2), b=np.ones(2) * 4)
        likelihood = multivariate_normal.med(sigma=np.eye(2))

        data = multivariate_normal.sample(mu=np.ones(2) * 2, sigma=np.eye(2), size=100)
        posterior = parameter_posterior(data, likelihood=likelihood, priors=prior, size=500, energies=0.05)

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        sb.kdeplot(data[:, 0], data[:, 1], label="data")
        plt.legend()

        plt.subplot(2, 1, 2)
        grid = 100
        x = np.linspace(0, 4.0, grid)
        y = np.linspace(-2.0, 6.0, grid)

        X, Y = np.meshgrid(x, y)
        Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = posterior.p(Z)

        plt.contourf(X, Y, Z.reshape(grid, grid))
        plt.show()

    def test_unilinear_normal_normal_prior(self):
        prior_variables = multivariate_normal.med(mu=np.zeros(2), sigma=np.eye(2) * 2)
        prior_noise = exponential.med(lam=1.0)
        likelihood = unilinear.med()

        x = np.linspace(-2, 2, 60)
        variables = np.array([2, 1])
        y = unilinear.sample(x=x, variables=variables, sigma=0.3)
        posterior = parameter_posterior((y, x),
                                        likelihood=likelihood,
                                        priors=(prior_variables, prior_noise),
                                        size=10000,
                                        batch=10,
                                        energies=0.05)

        print(posterior.sample(size=3000).mean(axis=0))

    def test_custom_logistic_regression(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def likelihood(y, x, w):
            return normal.p((y - sigmoid(x @ w[:, None, :-1] + w[:, None, None, -1]).squeeze(axis=2)),
                            mu=0.0, sigma=0.1)

        x = np.linspace(-5, 5, 50).reshape(-1, 1)
        y = (x > 0).astype(np.float).flatten()

        posterior = parameter_posterior((y, x),
                                        likelihood=likelihood,
                                        priors=multivariate_normal.med(mu=np.zeros(2), sigma=np.eye(2)),
                                        batch=5,
                                        size=10000)

        mean = posterior.sample(size=3000).mean(axis=0)

        print("accuracy", (y == np.round(sigmoid(x @ mean[:-1] + mean[-1]))).sum() / y.size)

        i = np.linspace(-2, 5, 100)
        j = np.linspace(-4, 4, 100)
        I, J = np.meshgrid(i, j)
        K = np.concatenate([I.reshape(-1, 1), J.reshape(-1, 1)], axis=1)
        K = posterior.p(K).reshape(100, 100)

        plt.figure(figsize=(10, 6))
        plt.title("Parameter distribution", fontsize=18)
        plt.contourf(I, J, K)
        plt.tight_layout()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig("../../images/custom_logistic_regression_example.png", bbox_inches="tight")
        plt.show()

    def test_normal_with_exponential_prior_mcmc_moment_matching(self):
        prior = exponential.med(lam=0.6)
        likelihood = normal.med(sigma=1.0)

        data = normal.sample(mu=3.0, sigma=2.0, size=200)
        posterior = parameter_posterior(data,
                                        likelihood=likelihood,
                                        priors=prior,
                                        size=10000,
                                        batch=25,
                                        match_moments_for=normal)

        print(posterior)

    def test_multinormal_with_uniform_prior_mcmc_moment_matching(self):
        prior = multivariate_uniform.med(a=np.ones(2) * -2, b=np.ones(2) * 2)
        likelihood = multivariate_normal.med(sigma=np.eye(2))
        data = multivariate_normal.sample(mu=np.ones(2) * 0, sigma=np.eye(2), size=300)

        posterior = parameter_posterior(data,
                                        likelihood=likelihood,
                                        priors=prior,
                                        size=20000,
                                        batch=25,
                                        match_moments_for=multivariate_normal)

        print(posterior)

    def test_sequential_custom(self):
        prior = multivariate_normal.med(mu=np.zeros(2), sigma=np.eye(2) * 2)

        def likelihood(y, x, w):
            result = []
            for _w in w: result.append(normal.p(y - x * _w[0] - _w[1], mu=0.0, sigma=0.3))
            return np.array(result)

        x = np.linspace(-1, 1, 100)
        y = x * 2 + 0.5 + normal.sample(mu=0.0, sigma=0.3, size=100)

        posterior = parameter_posterior((y, x),
                                        likelihood=likelihood,
                                        priors=prior,
                                        size=10000,
                                        batch=10,
                                        match_moments_for=multivariate_normal)

        print(posterior)


if __name__ == '__main__':
    unittest.main()
