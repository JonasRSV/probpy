from probpy.core import RandomVariable
import numpy as np
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


def _generic_from_likelihood_priors(data: Union[np.ndarray, Tuple[np.ndarray]],
                                    likelihood: RandomVariable,
                                    priors: Union[RandomVariable, Tuple[RandomVariable]]) -> RandomVariable:

    prior_sz = [prior.sample().size for prior in priors]
    n_priors = len(priors)

    arg_ranges = []
    accum = 0
    for i, _ in enumerate(prior_sz):
        if i == 0: arg_ranges.append((0, prior_sz[i]))
        else: arg_ranges.append((accum, accum + prior_sz[i]))

        accum += prior_sz[i]

    def _probability_data_tuple(x):
        args = [x[i:j] for i, j in arg_ranges]
        res = np.log(likelihood.p(*data, *args)).sum() \
               + np.sum([np.log(priors[i].p(args[i])) for i in range(n_priors)])
        return res

    def _probability_data(x):
        args = [x[i:j] for i, j in arg_ranges]
        return np.log(likelihood.p(data, *args)).sum() \
               + np.sum([np.log(priors[i].p(args[i])) for i in range(n_priors)])

    print(type(data))
    if type(data) == tuple: return generic.med(probability=_probability_data_tuple)
    return generic.med(probability=_probability_data)


def parameter_posterior(data: Union[np.ndarray, Tuple[np.ndarray]],
                        likelihood: RandomVariable,
                        priors: Union[RandomVariable, Tuple[RandomVariable]],
                        size: int = 1000,
                        energy: float = 0.05) -> RandomVariable:
    if type(priors) == RandomVariable: priors = (priors,)

    candidates = []
    if likelihood.cls in conjugates:
        candidates = conjugates[likelihood.cls]

    for conjugate in candidates:
        if len(priors) == 1 and conjugate.is_conjugate(likelihood, priors):
            return conjugate.posterior(data, likelihood, priors)

    rv = _generic_from_likelihood_priors(data, likelihood, priors)

    initial = np.concatenate([priors[i].sample().flatten() for i in range(len(priors))])
    samples = fast_metropolis_hastings_log_space(size=size, log_pdf=rv.p, initial=initial, energy=energy)

    return points.med(points=samples)
