from probpy.core import RandomVariable, Distribution
import numpy as np
from typing import Callable
from probpy.mcmc import fast_metropolis_hastings_log_space
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


def _generic_from_likelihood_priors(data: Union[np.ndarray, Tuple[np.ndarray]],
                                    likelihood: Callable[[Tuple[np.ndarray]], np.ndarray],
                                    priors: Union[RandomVariable, Tuple[RandomVariable]]) -> RandomVariable:
    epsilon = 1e-100
    prior_sz = [prior.sample().size for prior in priors]
    n_priors = len(priors)

    arg_ranges = []
    accum = 0
    for i, _ in enumerate(prior_sz):
        if i == 0:
            arg_ranges.append((0, prior_sz[i]))
        else:
            arg_ranges.append((accum, accum + prior_sz[i]))

        accum += prior_sz[i]

    def _probability_data_tuple(x):
        samples = x.shape[0]
        args = [x[:, i:j] for i, j in arg_ranges]
        prior_log_probability = np.sum([np.log(priors[i].p(args[i]) + epsilon).reshape(samples)
                                        for i in range(n_priors)], axis=0)
        data_log_probability = np.log(likelihood(*data, *args) + epsilon).sum(axis=1)
        data_log_probability = np.nan_to_num(data_log_probability, copy=False, nan=-10000.0)
        return prior_log_probability + data_log_probability

    def _probability_data(x):
        samples = x.shape[0]
        args = [x[:, i:j] for i, j in arg_ranges]
        prior_log_probability = np.sum([np.log(priors[i].p(args[i]) + epsilon).reshape(samples)
                                        for i in range(n_priors)], axis=0)
        data_log_probability = np.log(likelihood(data, *args) + epsilon).sum(axis=1)
        data_log_probability = np.nan_to_num(data_log_probability, copy=False, nan=-10000.0)

        return prior_log_probability + data_log_probability

    if type(data) == tuple: return generic.med(probability=_probability_data_tuple)
    return generic.med(probability=_probability_data)


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


def parameter_posterior(data: Union[np.ndarray, Tuple[np.ndarray]],
                        likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                        priors: Union[RandomVariable, Tuple[RandomVariable]],
                        size: int = 1000,
                        energy: float = 0.05,
                        parallel=25,
                        match_moments_for: Distribution = None) -> RandomVariable:
    if type(priors) == RandomVariable: priors = (priors,)

    if type(likelihood) == RandomVariable:
        conjugate = _attempt_conjugate(data, likelihood, priors)

        if conjugate is not None: return conjugate

    if type(likelihood) == RandomVariable:
        rv = _generic_from_likelihood_priors(data, likelihood.p, priors)
    else:
        rv = _generic_from_likelihood_priors(data, likelihood, priors)

    initial = np.concatenate([
        np.concatenate(
            [priors[i].sample().flatten() for i in range(len(priors))]).reshape(1, -1)
        for _ in range(parallel)
    ], axis=0)

    samples = fast_metropolis_hastings_log_space(size=size, log_pdf=rv.p, initial=initial, energy=energy)

    if match_moments_for is None:
        return points.med(points=samples)

    if match_moments_for in moment_matchers:
        return moment_matchers[match_moments_for](samples)
    else:
        raise NotImplementedError(f"No moment matcher implemented for {match_moments_for.__class__}")

