import unittest
import probpy as pp


class MyTestCase(unittest.TestCase):
    def test_something(self):

        coin = pp.bernoulli.med(probability=0.7)

        prior = pp.beta.med(a=1.0, b=1.0)  # 50 50 prior
        likelihood = pp.bernoulli.med()  # likelihood is a bernoulli

        for _ in range(20):
            outcome = coin.sample()
            print(outcome)
            prior = pp.parameter_posterior(outcome, likelihood=likelihood, priors=prior)

            prediction = pp.predictive_posterior(likelihood=likelihood, priors=prior)

            print(prediction.probability)


if __name__ == '__main__':
    unittest.main()
