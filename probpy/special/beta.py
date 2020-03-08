import numpy as np


#Taken from https://en.wikipedia.org/wiki/Beta_function
class Beta:

    @staticmethod
    def _beta_stirling(x, y):
        # This appears pretty bad for small beta values
        return np.sqrt(2 * np.pi) * np.float_power(x, x - 1 / 2) * np.float_power(y, y - 1 / 2) / np.float_power(x + y, x + y - 1 / 2)

    @staticmethod
    def _beta_todo(x, y):
        pass
        #TODO

    @staticmethod
    def beta(x, y):
        if x.shape != ():
            return np.array([Beta._beta_stirling(x_, y_) for x_, y_ in zip(x, y)])
        return Beta._beta_stirling(x, y)


