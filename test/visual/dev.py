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

        def logit(x):
            return np.log(x / (1 - x))

        student_skill = logit(0.7)

        items = logit(np.array([0.4, 0.6, 0.8, 0.7]))  # difficulties

        fast_p = pp.normal.fast_p

        def likelihood(obs, item, skill):  # IRT likelihood
            return fast_p(obs - sigmoid(skill - item), mu=0.0, sigma=0.6)

        ## IRT samples
        samples = 100
        obs, its = [], []
        for i in range(samples):
            item = items[np.random.randint(0, items.size)]
            outcome = float(np.random.rand() < sigmoid(student_skill - item))

            obs.append(outcome)
            its.append(item)


        prior_skill = pp.normal.med(mu=0.0, sigma=10)

        for i in range(samples)[:30]:
            print("obs i", obs[i], "its i", its[i])
            prior_skill = pp.parameter_posterior((obs[i], its[i]),
                                                 likelihood=likelihood, prior=prior_skill,
                                                 mode="search",
                                                 samples=300, batch=5,
                                                 volume=100, energy=0.1,
                                                 variance=2.0)

            mode = sigmoid(pp.mode(prior_skill)[0])

            print("observation", obs[i], "item", sigmoid(its[i]), "mode", mode)

if __name__ == '__main__':
    unittest.main()
