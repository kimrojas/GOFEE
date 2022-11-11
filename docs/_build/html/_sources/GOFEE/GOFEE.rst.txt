
=====
GOFEE
=====
Below an overview of the inputs to GOFEE is shown. Many of the inputs
can be safely kept at default settings. However ``calc``, defining the
calculator, as well as either ``startgenerator`` or ``structures``, 
defining the system, must always be set.
In addition it is recomended to also set the ``candidate_generatior``.

.. autoclass:: gofee.GOFEE
    :members: run


StartGenerator
==============
The :class:`StartGenerator` is used to generate initial structures for
the search. In addition it can be used to generate new candidates during
the search, if it is included as an operation in the
:class:`CandidateGenerator`.

.. autoclass:: gofee.candidates.StartGenerator
    :exclude-members:

CandidateGenerator
==================
The :class:`CandidateGenerator` is used in each iteration in the GOFEE search
to generate new candidates based on a list on operations to use. Possible
operation can be found below under the "Mutations" section.

.. autoclass:: gofee.candidates.CandidateGenerator
    :members: get_new_candidate


.. .. autoclass:: candidate_operations.CandidateGenerator
    :members: get_new_candidate

Mutations
=========
The mutations to chose from are listed below.

RattleMutation
--------------

.. autoclass:: gofee.candidates.RattleMutation
    :exclude-members:

RattleMutation2
---------------

.. autoclass:: gofee.candidates.RattleMutation2
    :exclude-members:

PermutationMutation
-------------------

.. autoclass:: gofee.candidates.PermutationMutation
    :exclude-members: