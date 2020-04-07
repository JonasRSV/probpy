
Distributions
------------------

Example:

.. code-block:: python

    import probpy as pp

    rv = pp.normal.med(mu=1.0, sigma=2.0)

    samples = rv.sample(size=100)
    densities = rv.p(samples)

    # conditional distribution

    rv = pp.normal.med(sigma=2.0)

    samples = rv.sample(mu=1.0, size=100)
    densities = rv.p(samples, mu=1.0)

.. automodule:: probpy.distributions.normal
    :members:

.. automodule:: probpy.distributions.bernoulli
    :members:

.. automodule:: probpy.distributions.beta
    :members:

.. automodule:: probpy.distributions.binomial
    :members:

.. automodule:: probpy.distributions.categorical
    :members:

.. automodule:: probpy.distributions.dirichlet
    :members:

.. automodule:: probpy.distributions.exponential
    :members:

.. automodule:: probpy.distributions.function
    :members:

.. automodule:: probpy.distributions.gamma
    :members:

.. automodule:: probpy.distributions.gaussian_process
    :members:

.. automodule:: probpy.distributions.generic
    :members:

.. automodule:: probpy.distributions.geometric
    :members:

.. automodule:: probpy.distributions.hypergeometric
    :members:

.. automodule:: probpy.distributions.multinomial
    :members:

.. automodule:: probpy.distributions.normal_inverse_gamma
    :members:

.. automodule:: probpy.distributions.points
    :members:

.. automodule:: probpy.distributions.poisson
    :members:

.. automodule:: probpy.distributions.uniform
    :members:

.. automodule:: probpy.distributions.unilinear
    :members:
