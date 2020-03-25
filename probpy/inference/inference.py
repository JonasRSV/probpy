from probpy.core import RandomVariable
from typing import Tuple
import numpy as np
from probpy.distributions import normal, multivariate_normal
from .conjugate import NormalNormal_MuPrior1D, MultivariateNormalNormal_MuPrior
from typing import Union, Tuple
from collections import defaultdict

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
    }
}


def posterior(data: np.ndarray,
              likelihood: RandomVariable,
              priors: Union[RandomVariable, Tuple[RandomVariable]]) -> RandomVariable:
    if type(priors) == RandomVariable: priors = (priors, )

    candidates = []
    if likelihood.cls in conjugates and len(priors) in conjugates[likelihood.cls]:
        candidates = conjugates[likelihood.cls][len(priors)]

    for conjugate in candidates:
        if conjugate.check(likelihood, priors):
            return conjugate.posterior(data, likelihood, priors)

    raise NotImplementedError("Non conjugate posteriors not implemented yet")
