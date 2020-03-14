import unittest

from probpy.integration import uniform_importance_sampling
from probpy.distributions import normal, multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


class TestIntegration(unittest.TestCase):
    def test_uniform_importance_sampling(self):
        f = lambda x: -np.square(x) + 3

        plt.figure(figsize=(10, 6))
        for sz in np.logspace(3, 5, 3):
            sz = int(sz)
            results = []
            for i in range(100):
                results.append(uniform_importance_sampling(size=sz,
                                                           function=f,
                                                           domain=(-2, 2),
                                                           proposal=normal.freeze(mu=0, sigma=2)))

            sb.distplot(results, label=f"size {sz}")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/uniform_importance_sampling.png", bbox_inches="tight")
        plt.show()

    def test_uniform_importance_multivariate(self):
        f = lambda x: -np.square(x[:, 0]) + np.square(x[:, 1])

        lower_bound = np.array([0, 0])
        upper_bound = np.array([4, 2])
        plt.figure(figsize=(10, 6))
        for sz in np.logspace(3, 5, 3):
            sz = int(sz)
            results = []
            for i in range(100):
                results.append(uniform_importance_sampling(size=sz,
                                                           function=f,
                                                           domain=(lower_bound, upper_bound),
                                                           proposal=multivariate_normal.freeze(mu=np.zeros(2), sigma=np.eye(2) * 2)))

            sb.distplot(results, label=f"size {sz}")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../images/uniform_importance_sampling_multivariate.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
