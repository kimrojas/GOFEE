.. _modify_gpr:

=========================
Modifying surrogate model
=========================

This tutorial extends the previous one for
:ref:`Cu15 clusters <cu_cluster_search>`. It is
therefore recomended that you do that one before the present one.

In the avove mentioned tutorial GOFEE was initialized with the following
arguments::

    from gofee import GOFEE
    search = GOFEE(calc=calc,
                   startgenerator=sg,
                   candidate_generator=candidate_generator,
                   max_steps=60,
                   population_size=5)

however GOFEE takes a number of other arguments, including a
Gaussian Process regression (GPR) model, which is actively learned
during the search and used for cheap optimization of new candidates.

One can for example apply a GPR model with another degree of regularization
in the search. This is controlled by the ``noise`` parameter of the ``kernel``,
passed to the GPR model. The modification can be achieved by::

    from gofee.surrogate import GPR
    from gofee.surrogate.kernel import DoubleGaussKernel

    kernel = DoubleGaussKernel(noise=1e-6)
    gpr = GPR(kernel=kernel)

    search = GOFEE(calc=calc,
                   gpr=gpr,
                   startgenerator=sg,
                   candidate_generator=candidate_generator,
                   max_steps=60,
                   population_size=5)

