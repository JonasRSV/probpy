import unittest

from probpy.distributions import *


class TestRandomVariable(unittest.TestCase):
    def test_random_variable(self):
        rv = normal.med(mu=2.0, sigma=1.0)

        self.assertEqual(rv.mu, 2.0)
        self.assertEqual(rv.sigma, 1.0)

        rv = normal.med(sigma=1.0)
        self.assertEqual(rv.sample(mu=1.0, size=3).size, 3)



if __name__ == '__main__':
    unittest.main()
