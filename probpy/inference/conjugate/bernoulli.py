from probpy.core import RandomVariable
from typing import Tuple
from probpy.distributions import beta, bernoulli
from probpy.inference.conjugate.identification import _check_no_none_parameters, _check_only_none_is


class BernoulliBeta_PPrior:
    """predictive conjugate for bernoulli likelihood with beta parameter prior"""

    @staticmethod
    def is_conjugate(likelihood: RandomVariable, priors: Tuple[RandomVariable]):
        if priors[0].cls is beta \
                and _check_no_none_parameters(priors[0]) \
                and _check_only_none_is(likelihood, [bernoulli.probability]):
            return True
        return False

    @staticmethod
    def posterior(_: RandomVariable, priors: Tuple[RandomVariable]) -> RandomVariable:
        prior = priors[0]

        a = prior.parameters[beta.a].value
        b = prior.parameters[beta.b].value

        return bernoulli.med(probability=(a / (a + b)))
