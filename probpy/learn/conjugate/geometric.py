from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import beta, geometric
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np


class GeometricBeta_PPrior:
    """Conjugate prior for geometric likelihood with unknown probability"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, prior: RandomVariable):
        if prior.cls is beta \
                and _check_no_none_parameters(prior) \
                and _check_only_none_is(likelihood, [geometric.probability]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, _: RandomVariable, prior: RandomVariable) -> RandomVariable:
        data = np.array(data[0])

        n = data.size

        prior_alpha = prior.parameters[beta.a].value
        prior_beta = prior.parameters[beta.b].value

        posterior_alpha = prior_alpha + n
        posterior_beta = prior_beta + data.sum()

        return beta.med(a=posterior_alpha, b=posterior_beta)
