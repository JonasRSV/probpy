from probpy.core import RandomVariable
from typing import Tuple
import numpy as np
from probpy.distributions import (normal,
                                  multivariate_normal,
                                  bernoulli,
                                  categorical,
                                  exponential)

from .conjugate import (NormalNormal_MuPrior1D,
                        MultivariateNormalNormal_MuPrior,
                        BernoulliBeta_PPrior,
                        CategoricalDirichlet_PPrior,
                        ExponentialGamma_LambdaPrior)
from typing import Union, Tuple

# (1) Likelihood distribution
# (2) number of priors
# (name of unknown parameters)

conjugates = {
    normal: {
        1: [
            NormalNormal_MuPrior1D
        ]
    },
    multivariate_normal: {
        1: [
            MultivariateNormalNormal_MuPrior
        ]
    },
    bernoulli: {
        1: [
            BernoulliBeta_PPrior
        ]
    },
    categorical: {
        1: [
            CategoricalDirichlet_PPrior
        ]
    },
    exponential: {
        1: [
           ExponentialGamma_LambdaPrior
        ]
    }


}


def posterior(data: np.ndarray,
              likelihood: RandomVariable,
              priors: Union[RandomVariable, Tuple[RandomVariable]]) -> RandomVariable:
    if type(priors) == RandomVariable: priors = (priors,)

    candidates = []
    if likelihood.cls in conjugates and len(priors) in conjugates[likelihood.cls]:
        candidates = conjugates[likelihood.cls][len(priors)]

    for conjugate in candidates:
        if conjugate.check(likelihood, priors):
            return conjugate.posterior(data, likelihood, priors)

    raise NotImplementedError("Non conjugate posteriors not implemented yet")
