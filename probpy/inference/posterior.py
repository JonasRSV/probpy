from probpy.core import RandomVariable
from typing import Union, Tuple, Callable
from probpy.integration import expected_value
import numpy as np

from probpy.distributions import (bernoulli, normal, multivariate_normal, generic)
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


def _attempt_conjugate(likelihood: RandomVariable,
                       priors: [RandomVariable]):
    candidates = []
    if likelihood.cls in conjugates:
        candidates = conjugates[likelihood.cls]

    for conjugate in candidates:
        if len(priors) == 1 and conjugate.is_conjugate(likelihood, priors):
            return conjugate.posterior(likelihood, priors)

    return None


def _generic_proposal_from_priors(priors: Tuple[RandomVariable]) -> RandomVariable:
    prior_sizes = [prior.sample().size for prior in priors]

    def _sample(size: int = ()):
        return np.concatenate([
            priors.sample(size=size).reshape(-1, prior_sizes[i])
            for i, prior in enumerate(priors)
        ], axis=1)

    argument_ranges = []
    j = 0
    for i, _ in enumerate(prior_sizes):
        if i == 0:
            argument_ranges.append((0, prior_sizes[i]))
        else:
            argument_ranges.append((j, j + prior_sizes[i]))

        j += prior_sizes[i]

    def _probability(x: np.ndarray):
        samples = x.shape[0]
        args = [x[:, i:j] for i, j in argument_ranges]
        probabilities = np.prod([priors.p(args[i]).reshape(samples) for i, prior in enumerate(priors)], axis=0)

        print(probabilities.shape)
        print(samples.shape)

        return probabilities

    return generic.med(sampling=_sample, probability=_probability)


def _integrate_probability(data: Tuple[np.ndarray],
                           likelihood: Callable[[Tuple[np.ndarray]], np.ndarray],
                           priors: Tuple[RandomVariable],
                           size: int) -> float:
    proposal = _generic_proposal_from_priors(priors)

    def _function(*args):
        return likelihood.p(*data, *args)

    return expected_value(size=size,
                          function=_function,
                          distribution=proposal)


def predictive_posterior(likelihood: Union[RandomVariable, Callable[[Tuple[np.ndarray]], np.ndarray]],
                         priors: Union[RandomVariable, Tuple[RandomVariable]],
                         data: Tuple[np.ndarray] = None,
                         size: int = 1000) -> Union[RandomVariable, float]:
    if type(likelihood) == RandomVariable:
        conjugate = _attempt_conjugate(likelihood, priors)
        if conjugate is not None: return conjugate

    if data is not None:
        return _integrate_probability(data, likelihood, priors, size)
    else:
        raise NotImplementedError("For non-conjugate non-data is not implemented yet")
