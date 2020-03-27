from probpy.core import RandomVariable
from typing import Union, Tuple

from probpy.distributions import (bernoulli, normal, multivariate_normal)
from .conjugate import (BernoulliBeta_PPrior, NormalNormal_MuPrior1D, MultivariateNormalNormal_MuPrior)

conjugates = {
    bernoulli: [
        BernoulliBeta_PPrior
    ],
    normal: [
        NormalNormal_MuPrior1D
    ],
    multivariate_normal: [
        MultivariateNormalNormal_MuPrior
    ]
}


def predictive_posterior(likelihood: RandomVariable,
                         priors: Union[RandomVariable, Tuple[RandomVariable]]):
    if type(priors) == RandomVariable: priors = (priors,)

    candidates = []
    if likelihood.cls in conjugates:
        candidates = conjugates[likelihood.cls]

    for conjugate in candidates:
        if conjugate.is_conjugate(likelihood, priors):
            return conjugate.posterior(likelihood, priors)

    raise NotImplementedError("Non conjugate posteriors not implemented yet")
