.. _au_on_cu_search:

=======================
Au7 on Cu(111) with EMT
=======================

In this tutorial we carry out a search for Au7-clusters on
the Cu(111) surface.

In this search we will utilize an :class:`OperationConstraint`
to constraint the :class:`RattleMutation` to only rattle the
atoms within a certain box in space that we define.

The following script :download:`Au7_on_Cu111.py` is used to carry
out the constrainted search:

.. literalinclude:: Au7_on_Cu111.py

And run using::

    mpiexec python Au7_on_Cu111.py

What follows is a description of the elements of the python code
above, which relates to consraining the atomix position during
the search.

Box setup
---------
As previously in the :ref:`Cu15 cluster search <cu_cluster_search>`,
we define a box in which initial atoms are placed. In this example we
will also use this box to constrain the position of the "free" atoms
during the search.
Defining a box positioned 0.3Å above the slab, with a height of 5Å and
with xy-dimensions shrunk, from all sides by a fraction "k", relative 
to the cell xy-dimensions, can be achieved by::

    ## Box for startgenerator and rattle-mutation
    k = 0.2  # Shrinkage fraction from each side of the box in v[0] and v[1] directions.
    cell = template.get_cell()
    # Initialize box with cell
    v = np.copy(cell)
    # Set height of box
    v[2][2] = 5
    # Shrink box in v[0] and v[1] directions
    v[0] *= (1-2*k)
    v[1] *= (1-2*k)
    # Chose anker point p0 so box in centered in v[0] and v[1] directions.
    z_max_slab = np.max(template.get_positions()[:,2])
    p0 = np.array((0, 0, z_max_slab+0.3)) + k*(cell[0]+cell[1])
    # Make box
    box = [p0, v]

Constraint object
-----------------
The constrraint object is made using::

    from gofee.utils import OperationConstraint
    box_constraint = OperationConstraint(box=box)

Initialize constrained GOFEE search
-----------------------------------
The constrained GOFEE search is initialized using the ``position_constraint``
keyword::

    from gofee import GOFEE
    search = GOFEE(calc=calc,
                startgenerator=sg,
                candidate_generator=candidate_generator,
                max_steps=150,
                population_size=5,
                position_constraint=box_constraint)