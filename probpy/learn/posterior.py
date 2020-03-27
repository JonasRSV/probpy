from probpy.core import RandomVariable
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
                                  unknown,
                                  unilinear)

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
                        UnknownGaussianProcess_FunctionPrior,
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
    unknown: [
        UnknownGaussianProcess_FunctionPrior
    ],
    unilinear: [
        UniLinearMultivariateNormal_VariablePrior
    ]
}


def parameter_posterior(data: np.ndarray,
                        likelihood: RandomVariable,
                        priors: Union[RandomVariable, Tuple[RandomVariable]]) -> RandomVariable:
    if type(priors) == RandomVariable: priors = (priors,)

    candidates = []
    if likelihood.cls in conjugates:
        candidates = conjugates[likelihood.cls]

    for conjugate in candidates:
        if conjugate.is_conjugate(likelihood, priors):
            return conjugate.posterior(data, likelihood, priors)

    raise NotImplementedError("Non conjugate posteriors not implemented yet")
