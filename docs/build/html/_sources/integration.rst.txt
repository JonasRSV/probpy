Integration
------------------

Example:

.. code-block:: python

    import probpy as pp
    import numpy as np

    f = lambda x: -np.square(x[:, 0]) + np.square(x[:, 1])

    lower_bound = np.array([0, 0])
    upper_bound = np.array([4, 2])

    proposal = pp.multivariate_normal.med(mu=np.zeros(2), sigma=np.eye(2) * 2)

    result = pp.uniform_importance_sampling(size=100000,
                                            function=f,
                                            domain=(lower_bound, upper_bound),
                                            proposal=proposal)


.. automodule:: probpy.integration
    :members:
