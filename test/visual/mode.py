import unittest
import numpy as np
import probpy as pp
import matplotlib.pyplot as plt


class VisualTestMode(unittest.TestCase):
    def test_mode_points(self):
        diag_mean = np.array([-1, 1])
        samples = np.concatenate([
            pp.multivariate_normal.sample(mu=np.ones(2) * -3, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=np.ones(2) * 3, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=np.ones(2) * 0, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=diag_mean * -3, sigma=np.eye(2), size=1000),
            pp.multivariate_normal.sample(mu=diag_mean * 3, sigma=np.eye(2), size=1000),
        ])

        distribution = pp.distributions.points.med(points=samples)


        print(pp.mode(distribution))

        x = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, x)
        Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = distribution.p(Z).reshape(100, 100)

        plt.figure(figsize=(10, 6))
        plt.contourf(X, Y, Z)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.tight_layout()
        plt.savefig("../../images/mode.png", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    unittest.main()
