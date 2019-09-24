#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --ntasks-per-node=36
#SBATCH --threads-per-core=1
#SBATCH --time=36:00:00

module purge
module load intel/18.2.199
module load intelmpi/18.2

export LD_LIBRARY_PATH=/usr/local/nlopt/2.6.1/lib64/:$LD_LIBRARY_PATH

EXE=$(pwd)/topopt

mkdir ~/tmpdir/Topopt-${SLURM_JOBID}
cd ~/tmpdir/Topopt-${SLURM_JOBID}

#srun $(placement $SLURM_NTASKS_PER_NODE 1) $EXE -filter 2 -rmin 0.16 -volfrac 0.1 -maxItr 200 -nx 177 -ny 177 -nz 353 > out.txt
srun -n 1 $EXE -malloc_debug -filter 2 -rmin 0.16 -volfrac 0.1 -maxItr 200 -nx 177 -ny 177 -nz 353 > out.txt

jobinfo ${SLURM_JOBID}


