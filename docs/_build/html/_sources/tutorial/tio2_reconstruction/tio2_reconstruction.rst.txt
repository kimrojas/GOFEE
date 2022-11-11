================================================
Searching for the TiO2(001)-(1x4) reconstruction
================================================

For this tutorial we will use the dftb-calculator with
the tiorg parameters. 

This tutorial is very similar to the previous one for TiO clusters,
:ref:`searching for TiO clusters <searching-for-TiO-clusters>`. It is
recomended that you do that one before the present one, as it is more
detailed.

The major difference in the present tutorial is that the template will
now not be empty, but contain a number of atoms fixed at bulk positions.

The template is defined in the file :download:`TiO2_slab.traj`. The
following code :download:`TiO2.py` is used to carry out the search:

.. literalinclude:: TiO2.py

And run using::

    mpiexec python TiO2.py

Setting up the system - atoms in template
=========================================

In this case the *template* contains a number of fixed atoms representing the
slap, on top of which we want to optimize a number of atoms given by
*stoichiometry*. The final thing we need to initialize the :class:`StartGenerator`
, used for generation initial structures, is the *box* within which the
:class:`StartGenerator` places atoms randomly.
In this case we choose a box=[p0, v] of height 2.5 starting at p0=(0,0,8), which
is slightly above the slab atoms.
To initialize the startgenerator, we first read in the template::

    from ase.io import read
    slab = read('TiO2_slab.traj', index='0')

then define the stoichiometry of atoms to be optimized on top of the slab,
in the form of a list of atomic numbers::

    stoichiometry = 5*[22]+10*[8]

Then define the *box* within which the :class:`StartGenerator` places atoms randomly::

    import numpy as np
    v = slab.get_cell()
    v[2,2] = 2.5
    p0 = np.array((0.0,0.,8.))
    box = [p0, v]

Finally the :class:`StartGenerator` can be initialized::

    from gofee.candidates import StartGenerator
    sg = StartGenerator(slab, stoichiometry, box)