.. _searching-for-TiO-clusters:

==========================
Searching for TiO clusters
==========================

For this tutorial we will use the dftb-calculator with
the tiorg parameters.

In this tutorial we carry out a search for titanium-oxide clusters using
dftb to evaluate energies and forses of the structures.

The following script :download:`Ti5O10.py` is used to carry out the search (the indivitual elements are
explainted further below):

.. literalinclude:: Ti5O10.py

And run using::

    mpiexec python Ti5O10.py

What follows is a description of the python script above.

Setting up the system
=====================

An important prerequisite for starting a search is to set up the system.
This is done by defining a template and a stoichiometry of the atoms to
optimize.

The *template* is an :class:`Atoms` object, either describing an empty cell or
a cell containing for example a slab of atoms. For most purposes, the atoms
in the template shold be fixed using :class:`ase.constraints.FixAtoms`
constraint, as the template atoms are kept fixed during mutation operation,
but will take part in surrogate-relaxation if not fixed.
In this example the template is taken to be an empty 20Åx20Åx20Å cell, since
we considder isolated TiO-clusters. The code to generate the template is:: 

    from ase import Atoms
    template = Atoms('',
                cell=[20,20,20],
                pbc=[0, 0, 0])

The *stoichiometry* of atoms to optimize is a list of atomic numbers. In this
case 5 titanium (atomic nymber 22) and 10 oxygen (atomic number 8) atoms::

    stoichiometry = 5*[22]+10*[8]

Startgenerater - for making initial structures
==============================================

To initialize the search, initial structures need to be generated. This is
carried out using the :class:`StartGenerator`, which in addition to the
*template* and *stoichiometry* defined above, need a *box* in which to randomly
place the atoms defined in the *stoichiometry*.

The *box* is naturally defined by a point *p0* and three spanning vectors going
out from that point. These are defined bu the 3x3 matrix *v* in the example.
In the example a 20Åx20Åx20Å square box in the center of the cell is used::

    import numpy as np
    v = 4*np.eye(3)
    p0 = np.array((8.0, 8.0, 8.0))
    box = [p0, v]

The *startgenerator* can then be initialized with the code::

    from candidate_operations.candidate_generation import StartGenerator
    sg = StartGenerator(template, stoichiometry, box)

CandidateGenerator
==================

In GOFEE, the configurational space is explored by generation new candidate structures.
New candidates can be either completely random structures made using the *startgenerator*
or they can be the result of applying mutation operations to a population of some of the
best structures visited during the search. Examples of mutaion operations are the
:class:'RattleMutation', which randomly shifts some of the atoms and the
:class:`PermutaionMutation` which randomly permutes some atoms of different type.
The rattle mutation in the example, which rattles on average Natoms=3 atom a maximum distance of
rattle_range=4Å, is initialized as::

    from candidate_operations.basic_mutations import RattleMutation
    n_to_optimize = len(stoichiometry)
    rattle = RattleMutation(n_to_optimize, Nrattle=3, rattle_range=4)

Given some of the above described operations. e.g. a :class:`StartGenerator`
and a :class:'RattleMutation', one can initialize a :class:`CandidateGenerator`,
which handles the generation of new candidates by applying the supplied
*operations* with probability specified in the *probabilities* list.
A CandidateGenerator which uses the startgenerator *sg* with 20% probability and
the rattle operation *rattle* with 80% probability, is initialized as follows::

    from candidate_operations.candidate_generation import CandidateGenerator
    candidate_generator = CandidateGenerator(probabilities=[0.2, 0.8],
                                             operations=[sg, rattle])

Initialize and run GOFEE
========================

With all the above objects defined, we are ready to initialize and run GOFEE.
To run the search for 100 iterations with a population size of 5, use::

    from gofee import GOFEE
    search = GOFEE(calc=calc,
                   startgenerator=sg,
                   candidate_generator=candidate_generator,
                   max_steps=100,
                   population_size=5)
    search.run()

This tutorial relies on many default settings of GOFEE, which could be changed.
To see how these settings are changed, have a look at the other tutorials.

