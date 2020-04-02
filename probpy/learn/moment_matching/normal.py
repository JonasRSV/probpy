import numpy as np
from probpy.distributions import normal, multivariate_normal


def univariate_normal_matcher(samples: np.ndarray):
    if type(samples) == list: samples = np.concatenate(samples, axis=1)
    mean = samples.mean()
    variance = samples.var()
    return normal.med(mu=mean, sigma=variance)


def multivariate_normal_matcher(samples: np.ndarray):
    if type(samples) == list: samples = np.concatenate(samples, axis=1)
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False)
    return multivariate_normal.med(mu=mean, sigma=covariance)