from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import beta, binomial
from .identification import _check_no_none_parameters, _check_only_none_is
import numpy as np


class BinomialBeta_PPrior:
    """Conjugate prior for binomial likelihood with unknown probability"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, prior: RandomVariable):
        if prior.cls is beta \
                and _check_no_none_parameters(prior) \
                and _check_only_none_is(likelihood, [binomial.probability]):
            return True
        return False

    @staticmethod
    def posterior(data: np.ndarray, likelihood: RandomVariable, prior: RandomVariable) -> RandomVariable:
        data = np.array(data[0])

        n_data = data.size
        n = likelihood.parameters[binomial.n].value

        prior_alpha = prior.parameters[beta.a].value
        prior_beta = prior.parameters[beta.b].value

        posterior_alpha = prior_alpha + data.sum()
        posterior_beta = prior_beta + n_data * n - data.sum()

        return beta.med(a=posterior_alpha, b=posterior_beta)
