from probpy.core import RandomVariable
from typing import Union, Tuple, Callable
import numpy as np

from probpy.distributions import (normal,
                                  multivariate_normal,
                                  bernoulli,
                                  categorical,
                                  exponential,
                                  binomial,
                                  multinomial,
                                  poisson,
                                  geometric,
                                  unilinear)

from . import (NormalNormal_MuPrior1D,
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


def attempt(
        data: Union[Tuple[np.ndarray]],
        likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
        prior: RandomVariable
) -> RandomVariable:
    if type(likelihood) == RandomVariable:
        candidates = []
        if likelihood.cls in conjugates:
            candidates = conjugates[likelihood.cls]

        for conjugate in candidates:
            if conjugate.is_conjugate(likelihood, prior):
                return conjugate.posterior(data, likelihood, prior)

    return None
