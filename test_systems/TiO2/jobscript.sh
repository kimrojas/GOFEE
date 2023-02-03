#!/bin/bash
#$ -q xs2.q
#$ -pe x16 16
#$ -j y
#$ -cwd
#$ -S /bin/bash
#$ -N gofeeerf300
#$ -t 1-20:1

I_MPI_PIN=1
I_MPI_ADJUST_ALLGATHERV=2
OMP_NUM_THREADS=1

# -- ENVIRONMENT --
module load intel/2020.2.254 intelmpi/2020.2.254
export PYTHONPATH=/home/SHenninger/gofee_setup/gofee:$PYTHONPATH
mamba activate gofee
export PATH=$PATH:/home/msamuel/apps/dftbplus-21.2.x86_64-linux/bin/
export DFTB_PREFIX=/home/krojas/share/lib/slakos/tiorg
# mymamba
# mamba activate share_agoxenv

# -- MAIN PROCESS --
id=${SGE_TASK_ID}
id="$(($id-1))"
dir="runs0/run$id"
mkdir -p $dir
cp TiO2.py ${dir}
cp kappachanginggofee.py ${dir}
cp TiO2_slab.traj ${dir}
cd ${dir}
mpiexec python3 TiO2.py -s $(( $SGE_TASK_ID - 1 )) >> file.out
# -- MAIN PROCESS --


rm -f hostfile.$JOB_ID 
