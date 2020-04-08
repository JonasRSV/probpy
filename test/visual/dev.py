import unittest
import probpy as pp
import numpy as np
import random
import time


class MyTestCase(unittest.TestCase):
    def test_something_else(self):
        import probpy as pp
        import numpy as np

        f = lambda x: -np.square(x[:, 0]) + np.square(x[:, 1])

        lower_bound = np.array([0, 0])
        upper_bound = np.array([4, 2])

        proposal = pp.multivariate_normal.med(mu=np.zeros(2), sigma=np.eye(2) * 2)

        result = pp.uniform_importance_sampling(size=100000,
                                                function=f,
                                                domain=(lower_bound, upper_bound),
                                                proposal=proposal)

    def test_something(self):
        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))

        def logit(x):
            return np.log(x / (1 - x))

        student_skill = logit(0.7)

        items = logit(np.array([0.4, 0.6, 0.8, 0.7]))  # difficulties

        def likelihood(obs, item, skill):
            result = []
            for _skill in skill:
                result.append(pp.normal.p(obs - sigmoid(_skill - item), mu=0.0, sigma=0.6))

            return np.array(result)

        samples = 30
        obs, its = [], []
        for i in range(samples):  # 100 samples
            item = items[np.random.randint(0, items.size)]
            outcome = (np.random.rand() < sigmoid(student_skill - item)).astype(np.float)

            obs.append(outcome)
            its.append(item)

        prior_skill = pp.normal.med(mu=0.0, sigma=10)

        for i in range(samples):

            prior_skill = pp.parameter_posterior((obs[i], its[i]),
                                                 likelihood=likelihood, priors=prior_skill,
                                                 mode="search",
                                                 samples=500, batch=10,
                                                 volume=100,
                                                 variance=2.0)

            modes = sigmoid(pp.mode(prior_skill))

            print("obs", obs[i], "its", sigmoid(its[i]), "modes", modes)
            print()


if __name__ == '__main__':
    unittest.main()
