.. _slurm:

==================
SLURM job examples
==================

If you are running your calculations on a cluster using
SLURM for job management, you can use a jobscript like
this (propperly modified for your setup)::

    #!/bin/bash
    #SBATCH --job-name=GOFEE_test
    #SBATCH --partition=<desired partitions>
    #SBATCH --mem=30G
    #SBATCH --nodes=1
    #SBATCH --time=2:00:00
    #SBATCH --ntasks-per-node=10
    #SBATCH --cpus-per-task=1

    echo "========= Job started  at `date` =========="
    # Go to the directory where this job was submitted
    cd $SLURM_SUBMIT_DIR
    
    export PYTHONPATH=<path to GOFEE code>:$PYTHONPATH
    source <python stuff>
    source <DFTB stuff>  # if running DFTB

    mpiexec python run_search.py
    echo "========= Job finished at `date` =========="

This job will be run locally in the submission folder on 10 cpu cores.

NOTE: for this jobscript, the submission folder must contain a file
"run_search.py", which runs GOFEE as described in the
:ref:`tutorial <tutorial>`.