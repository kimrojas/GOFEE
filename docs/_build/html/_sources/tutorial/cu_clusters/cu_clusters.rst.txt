.. _cu_cluster_search:

========================
Cu15 with EMT - Detailed
========================

In this tutorial we carry out a search for isolated Cu15-clusters
described by the EMT potential for efficiency.

The following script :download:`Cu15.py` is used to carry out the search
(the indivitual elements of the code are explainted further below):

.. literalinclude:: Cu15.py

And run using::

    mpiexec python Cu15.py

What follows is a description of the python script above.

Setting up the system
=====================

An important prerequisite for starting a search is to set up the system.
This is done by defining a template and a stoichiometry of the atoms to
optimize.

The *template* is an :class:`Atoms` object, either describing an empty cell or
a cell containing for example a slab of atoms. For most purposes, the atoms
in the template shold be fixed using the :class:`ase.constraints.FixAtoms`
constraint, as the template atoms are kept fixed during mutation operation,
but will take part in surrogate-relaxation if not fixed.
In this example the template is taken to be an empty 20Åx20Åx20Å cell, since
we considder isolated Cu-clusters. The code to generate the template is:: 

    from ase import Atoms
    template = Atoms('',
                cell=[20,20,20],
                pbc=[0, 0, 0])

The *stoichiometry* of atoms to optimize is a list of atomic numbers. In this
case 15 copper atoms (atomic nymber 29)::

    stoichiometry = 15*[29]

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
    v = 5*np.eye(3)
    p0 = np.array((7.5, 7.5, 7.5))
    box = [p0, v]

The *startgenerator* can then be initialized with the code::

    from gofee.candidates import StartGenerator
    sg = StartGenerator(template, stoichiometry, box)

CandidateGenerator
==================

In GOFEE, the configurational space is explored by generation new candidate structures.
New candidates can be either completely random structures made using the *startgenerator*
or they can be the result of applying mutation operations to a population of some of the
best structures visited during the search. Examples of mutaion operations are the
:class:`RattleMutation`, which randomly shifts some of the atoms and the
:class:`PermutaionMutation` which randomly permutes some atoms of different type.
In this example we only optimize atoms of a single type, and therfor only use the
:class:`RattleMutation`, initializing it to rattle on average Natoms=3 atoms a maximum
distance of rattle_range=4Å, is achieved with::

    from gofee.candidates import RattleMutation
    n_to_optimize = len(stoichiometry)
    rattle = RattleMutation(n_to_optimize, Nrattle=3, rattle_range=4)

Given some of the above described operations. e.g. a :class:`StartGenerator`
and a :class:`RattleMutation`, one can initialize a :class:`CandidateGenerator`,
which handles the generation of new candidates by applying the supplied
*operations* with probability specified in the *probabilities* list.
A CandidateGenerator which uses the startgenerator *sg* with 20% probability and
the rattle operation *rattle* with 80% probability, is initialized as follows::

    from gofee.candidates import CandidateGenerator
    candidate_generator = CandidateGenerator(probabilities=[0.2, 0.8],
                                             operations=[sg, rattle])

Initialize and run GOFEE
========================

With all the above objects defined, we are ready to initialize and run GOFEE.
To run the search for 60 iterations with a population size of 5, use::

    from gofee import GOFEE
    search = GOFEE(calc=calc,
                startgenerator=sg,
                candidate_generator=candidate_generator,
                max_steps=60,
                population_size=5)
    search.run()

This tutorial relies on many default settings of GOFEE, which could be changed.
To see how these settings are changed, have a look at the other tutorials.

