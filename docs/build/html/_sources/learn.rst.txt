Learn
------------------

Example:

.. code-block:: python

    import probpy as pp

    prior = pp.normal.med(mu=1.0, sigma=2.0)
    likelihood = pp.normal.med(sigma=1.0)

    data = pp.normal.sample(mu=5.0, sigma=2.0, size=1000)

    prior = pp.parameter_posterior(data, likelihood=likelihood, priors=prior)

.. automodule:: probpy.learn.posterior.posterior
    :members:


.. automodule:: probpy.learn.posterior.mcmc
    :members:


.. automodule:: probpy.learn.posterior.search
    :members:
