import numpy as np
from probpy.special import gamma


#Taken from https://en.wikipedia.org/wiki/Beta_function
class Beta:

    @staticmethod
    def _beta_stirling(x, y):
        # This appears pretty bad for small beta values
        return np.sqrt(2 * np.pi) * np.float_power(x, x - 1 / 2) * np.float_power(y, y - 1 / 2) / np.float_power(x + y, x + y - 1 / 2)

    @staticmethod
    def _beta_from_gamma(x, y):
        return gamma(x) * gamma(y) / gamma(x + y)

    @staticmethod
    def beta(x, y):
        if x.shape != ():
            return np.array([Beta._beta_from_gamma(x_, y_) for x_, y_ in zip(x, y)])
        return Beta._beta_from_gamma(x, y)


