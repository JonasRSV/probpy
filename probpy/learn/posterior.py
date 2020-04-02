from probpy.core import RandomVariable, Distribution
import numpy as np
from typing import Callable, List
from probpy.mcmc import fast_metropolis_hastings_log_space_parameter_posterior_estimation
from probpy.distributions import (normal,
                                  multivariate_normal,
                                  bernoulli,
                                  categorical,
                                  exponential,
                                  binomial,
                                  multinomial,
                                  poisson,
                                  geometric,
                                  unilinear,
                                  generic,
                                  points)

from .conjugate import (NormalNormal_MuPrior1D,
                        NormalNormal_NormalInverseGammaPrior1D,
                        MultivariateNormalNormal_MuPrior,
                        BernoulliBeta_PPrior,
                        CategoricalDirichlet_PPrior,
                        ExponentialGamma_LambdaPrior,
                        BinomialBeta_PPrior,
                        MultinomialDirichlet_PPrior,
                        PoissonGamma_LambdaPrior,
                        GeometricBeta_PPrior,
                        UniLinearMultivariateNormal_VariablePrior)

from .moment_matching import (multivariate_normal_matcher, univariate_normal_matcher)
from typing import Union, Tuple

conjugates = {
    normal: [
        NormalNormal_MuPrior1D,
        NormalNormal_NormalInverseGammaPrior1D
    ],
    multivariate_normal: [
        MultivariateNormalNormal_MuPrior
    ],
    bernoulli: [
        BernoulliBeta_PPrior
    ],
    categorical: [
        CategoricalDirichlet_PPrior
    ],
    exponential: [
        ExponentialGamma_LambdaPrior
    ],
    binomial: [
        BinomialBeta_PPrior
    ],
    multinomial: [
        MultinomialDirichlet_PPrior
    ],
    poisson: [
        PoissonGamma_LambdaPrior
    ],
    geometric: [
        GeometricBeta_PPrior
    ],
    unilinear: [
        UniLinearMultivariateNormal_VariablePrior
    ]
}

moment_matchers = {
    normal: univariate_normal_matcher,
    multivariate_normal: multivariate_normal_matcher
}


def log_probabilities(data: Union[np.ndarray, Tuple[np.ndarray]],
                      likelihood: Callable[[Tuple[np.ndarray]], np.ndarray],
                      priors: Tuple[RandomVariable]):
    def _log_likelihood(*args):
        return np.nan_to_num(np.log(likelihood(*data, *args)).sum(axis=1), copy=False, nan=-100000,
                             neginf=-1000000, posinf=-1000000)

    log_priors = []
    for prior in priors:
        def _log_prior(x, prior=prior):
            return np.log(prior.p(x))

        log_priors.append(_log_prior)

    return _log_likelihood, log_priors


def _attempt_conjugate(data: Union[np.ndarray, Tuple[np.ndarray]],
                       likelihood: RandomVariable,
                       priors: [RandomVariable]):
    candidates = []
    if likelihood.cls in conjugates:
        candidates = conjugates[likelihood.cls]

    for conjugate in candidates:
        if len(priors) == 1 and conjugate.is_conjugate(likelihood, priors):
            return conjugate.posterior(data, likelihood, priors)

    return None


def _reshape_samples(samples: List[np.ndarray]):
    result = []
    for sample in samples:
        if sample.ndim == 1:
            result.append(sample.reshape(-1, 1))
        else:
            result.append(sample)
    return result


def parameter_posterior(data: Union[np.ndarray, Tuple[np.ndarray]],
                        likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                        priors: Union[RandomVariable, Tuple[RandomVariable]],
                        size: int = 1000,
                        energies: Tuple[float] = 0.05,
                        batch=10,
                        match_moments_for: Union[Tuple[Distribution], Distribution] = None) -> RandomVariable:
    if type(priors) == RandomVariable: priors = (priors,)
    if type(data) == np.ndarray: data = (data,)
    if type(energies) == float: energies = [energies for _ in range(len(priors))]

    if type(likelihood) == RandomVariable:
        conjugate = _attempt_conjugate(data, likelihood, priors)

        if conjugate is not None: return conjugate

    likelihood = likelihood if type(likelihood) != RandomVariable else likelihood.p

    log_likelihood, log_priors = log_probabilities(data, likelihood, priors)

    initial = [
        prior.sample(size=batch)
        for prior in priors
    ]

    samples = fast_metropolis_hastings_log_space_parameter_posterior_estimation(
        size=size,
        log_likelihood=log_likelihood,
        log_priors=log_priors,
        initial=initial,
        energies=energies)

    samples = _reshape_samples(samples)

    if match_moments_for is None:
        return points.med(points=np.concatenate(samples, axis=1))

    if type(match_moments_for) == tuple:
        matches = []
        for d, s in zip(match_moments_for, samples):
            matches.append(
                moment_matchers[d](s)
            )
        return matches
    else:
        return moment_matchers[match_moments_for](samples)
