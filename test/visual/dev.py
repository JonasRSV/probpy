import unittest
import probpy as pp
import numpy as np
import random


class MyTestCase(unittest.TestCase):
    def test_something_else(self):
        point = np.random.normal(size=200)

        p = pp.points.med(points=point)

    def test_something(self):
        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))

        def predict(w, x):
            return x[:, 0] * w[0] + x[:, 1] * w[1] + x[:, 2] * w[2] + w[3]

        w = [-3, 3, 5, -3]  # True underlying model

        x = np.random.rand(100, 3)
        y = sigmoid(predict(w, x) + pp.normal.sample(mu=0.0, sigma=1.0, size=100).reshape(-1))

        # For this we need custom likelihood since there is no conjugate prior

        def likelihood(y, x, w):
            return pp.normal.p((y - sigmoid(x @ w[:, :-1, None] + w[:, None, None, -1]).squeeze(axis=2)),
                               mu=0.0, sigma=1.0)

        prior = pp.multivariate_normal.med(mu=np.zeros(4), sigma=np.eye(4) * 5)

        for i in range(50):
            j = random.randint(0, 80)
            data = (y[j: j + 50], x[j: j + 50])

            prior = pp.parameter_posterior(data, likelihood=likelihood,
                                           priors=prior,
                                           batch=100,
                                           samples=10000,
                                           burn_in=1000,
                                           energies=1.0,
                                           normalize=False,
                                           mcmc=False)

            if i % 1 == 0:
                modes = pp.mode(prior)

                print("modes", len(modes))
                w_approx = modes[0]

                print("Parameter approximation", w_approx)
                print("Prior MSE", np.square(y - sigmoid(predict(w_approx, x))).mean(), "True MSE",
                      np.square(y - sigmoid(predict(w, x))).mean())
                print()


if __name__ == '__main__':
    unittest.main()
