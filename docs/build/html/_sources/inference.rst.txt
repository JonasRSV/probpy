Inference
------------------

Example:

.. code-block:: python

    import probpy as pp
    ## getting modes
    prior = pp.normal.med(mu=1.0, sigma=2.0)
    modes = pp.mode(prior)


.. automodule:: probpy.inference.mode
    :members:

