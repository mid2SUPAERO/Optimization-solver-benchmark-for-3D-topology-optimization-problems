#!/bin/bash
#SBATCH --nodes=40
#SBATCH --ntasks=800
#SBATCH --ntasks-per-node=20
#SBATCH --threads-per-core=1
#SBATCH -J Topopt1
#SBATCH --time=36:00:00
#SBATCH --mail-user=godfred.agyekum-oheneba@isae-supaero.fr

module purge
module load intel/18.2.199
module load intelmpi/18.2

EXE=$HOME/Topopt/topopt


mkdir /tmpdir/agyekum/Topopt-${SLURM_JOBID}
cd /tmpdir/agyekum/Topopt-${SLURM_JOBID}
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

srun $(placement $SLURM_NTASKS_PER_NODE 1) $EXE -filter 2 -rmin 0.16 -volfrac 0.1 -maxItr 200 -nx 257 -ny 257 -nz 513 > out.txt

jobinfo ${SLURM_JOBID}
infoincidentjob
