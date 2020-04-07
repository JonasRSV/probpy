Sampling
------------------

Example:

.. code-block:: python

    import probpy as pp
    import numpy as np

    # gaussian mixture pdf
    pdf = lambda x: pp.normal.p(x, 0, 1) + pp.normal.p(x, 6, 3) + pp.normal.p(x, -6, 0.5)

    samples = pp.metropolis_hastings(size=50000, 
                                     pdf=pdf, 
                                     proposal=pp.normal.med(sigma=1.0), 
                                     initial=-5)

    # 100x faster but does not take custom proposal
    samples = pp.fast_metropolis_hastings(size=50000, 
                                          pdf=pdf,
                                          initial=np.random.rand(100),
                                          energy=1.0)



.. automodule:: probpy.sampling.mcmc
    :members:

