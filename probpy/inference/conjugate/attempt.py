from probpy.core import RandomVariable
from probpy.distributions import (bernoulli, normal, multivariate_normal)
from .bernoulli import BernoulliBeta_PPrior
from .normal import NormalNormal_MuPrior1D, MultivariateNormalNormal_MuPrior
from typing import Union, Tuple, Callable

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


def attempt(likelihood: Union[RandomVariable, Callable],
            priors: Tuple[RandomVariable]):

    if type(likelihood) == RandomVariable:
        candidates = []
        if likelihood.cls in conjugates:
            candidates = conjugates[likelihood.cls]

        for conjugate in candidates:
            if len(priors) == 1 and conjugate.is_conjugate(likelihood, priors):
                return conjugate.posterior(likelihood, priors)

    return None
