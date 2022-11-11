.. _cu_au_cluster_search:

==============================
Cu3Au2 with EMT - First search
==============================

In this tutorial we carry out a search for isolated Cu3Au2-clusters
described by the EMT potential for efficiency. This simple system can
conveniently be run in a few minuts without the need for paralization.
A more detailed walkthrough is given in the 
:ref:`Cu15 tutorial <cu_cluster_search>`.

The following script :download:`Cu3Au2.py` is used to carry out the search:

.. literalinclude:: Cu3Au2.py

And run using::

    python Cu3Au2.py

Or to run in parallel::

    mpiexec python Cu3Au2.py

The setting::

    logfile='-'

Is used to write the search log to the standard output. Use::

    logfile='filename'

to write to a file.

Continue with the :ref:`Cu15 tutorial <cu_cluster_search>` for more detail.