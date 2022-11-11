.. _installation:

============
Installation
============

Most code is written in python, however timecritical parts such as descriptor
and prior-mean evaluations are implemented using cython (and cymem), and need
to be compiled for your particular setup.

Requirements
------------

* Python (tested with 3.6.3 and newer)
* ASE (tested with 3.17 and newer)
* Cython (tested with 0.28 and newer)
* cymem (tested with 1.31.2 and newer)
* mpi4py (tested with 3.0 and newer)

Install from source
-------------------

The code is avaliable as a tar-file :download:`gofee_stable.tar.gz` or 
:download:`gofee.tar.gz` for a newer, less tested version.

After downloading the tar-file, e.g. using "wget", unpack it using::

    tar -zxvf gofee.tar.gz

Then run the build_code file inside the gofee-folder, to compile descriptor
and prior-function, both used in the surrogate model. Do this using::

    ./build_code

This will compile the mentioned files for the python setup used
at the time of compiling.

Finally when using the code, you need to have the gofee-folder in
the PYTHONPATH. This is achieved using::

    export PYTHONPATH=<path-to-folder>/gofee:$PYTHONPATH

When this is done, you can run serial GOFEE sxripts using::

    python script_calling_GOFEE.py

and parallel ones using::

    mpiexec python script_calling_GOFEE.py

or if running with an older version of GPAW::

    mpiexec --mca mpi_warn_on_fork 0 gpaw-python script_calling_GOFEE.py
