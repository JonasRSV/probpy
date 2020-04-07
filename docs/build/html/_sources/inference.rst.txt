Inference
------------------

Example:

.. code-block:: python

    import probpy as pp

    prior = pp.normal.med(mu=1.0, sigma=2.0)
    likelihood = pp.normal.med(sigma=1.0)

    posterior = pp.predictive_posterior(likelihood=likelihood, priors=prior)

    ## Or just getting modes
    prior = pp.normal.med(mu=1.0, sigma=2.0)

    modes = pp.mode(prior)


.. automodule:: probpy.inference.mode
    :members:

.. automodule:: probpy.inference.posterior
    :members:
