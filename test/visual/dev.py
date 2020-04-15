import unittest
import probpy as pp
import numpy as np
import numba
import random
import time
import os


class MyTestCase(unittest.TestCase):
    def test_something_else(self):
        @numba.jit(nopython=True, fastmath=True, forceobj=False)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        x = np.linspace(-5, 5, 50).reshape(-1, 1)
        y = (x > 0).astype(np.float).flatten()

        def likelihood_old(y, x, w):
            return pp.normal.p((y - sigmoid(x @ w[:, None, :-1] + w[:, None, None, -1]).squeeze(axis=2)),
                               mu=0.0, sigma=0.1)

        p = pp.normal.fast_p

        @numba.jit(nopython=True, fastmath=True, forceobj=False)
        def likelihood_broadcast(y, x, w):
            return p(y - sigmoid(x * w[0] + w[1]), mu=0.0, sigma=0.1)

        @numba.jit(nopython=True, fastmath=True, forceobj=False)
        def loop_broadcast(y, x, w):
            return [likelihood_broadcast(y, x, w[i]) for i in range(len(w))]

        @numba.jit(nopython=True, fastmath=True, forceobj=False)
        def likelihood(y, x, w):
            return p(y - sigmoid(x * w[0] + w[1]), mu=0.0, sigma=0.1)[0]

        @numba.jit(nopython=True, fastmath=True, forceobj=False, parallel=True)
        def large_loop_broadcast(y, x, w):
            result = np.zeros(len(w))
            for i in range(result.size):
                for j in range(y.size):
                    result[i] += np.log(likelihood(y[i], x[i], w[j]))

            return result

        w = np.random.rand(100, 2)

        loop_broadcast(y, x, w)
        large_loop_broadcast(y, x, w)

        timestamp = time.time()
        old = likelihood_old(y, x, w)
        print("Old ", time.time() - timestamp)

        timestamp = time.time()
        loop = loop_broadcast(y, x, w)
        print("loop ", time.time() - timestamp)

        timestamp = time.time()
        large_loop = large_loop_broadcast(y, x, w)
        print("large loop ", time.time() - timestamp)

        print("old", old)
        # print("loop", loop)
        print("large loop", np.array(large_loop))

    def test_something_else_stuff(self):
        @numba.jit(nopython=True, fastmath=True, forceobj=False)
        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))

        fast_p = pp.normal.fast_p  # Need to assign here first since numba does not support jitting methods of classes

        def likelihood(y, x, w):
            return fast_p(y - sigmoid(np.sum(x * w[:-1]) + w[-1]), mu=0.0, sigma=1.0)

        def predict(w, x):
            return x[:, 0] * w[0] + x[:, 1] * w[1] + x[:, 2] * w[2] + w[3]


        w = [-3, 3, 5, -3]  # True underlying model

        x = np.random.rand(100, 3)
        y = sigmoid(predict(w, x) + pp.normal.sample(mu=0.0, sigma=1.0, size=100).reshape(-1))

        prior = pp.multivariate_normal.med(mu=np.zeros(4), sigma=np.eye(4) * 10)

        for i in range(5):
            data = (y, x)

            prior = pp.parameter_posterior(data, likelihood=likelihood,
                                           prior=prior,
                                           batch=50,
                                           samples=1000,
                                           energy=0.25,
                                           mode="search",
                                           volume=1000)

            modes = pp.mode(prior)  # modes are sorted in order first is largest

            print("Number of modes", len(modes))
            for mode in modes:
                print(mode)

            w_approx = modes[0]

            print("Parameter Estimate", w_approx)

            print("Prior MSE", np.square(y - sigmoid(predict(w_approx, x))).mean(),
                  "True MSE", np.square(y - sigmoid(predict(w, x))).mean())
            print()


if __name__ == '__main__':
    unittest.main()
