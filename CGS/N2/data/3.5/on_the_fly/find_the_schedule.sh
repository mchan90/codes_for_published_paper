#!/bin/bash
#
#SBATCH --job-name=3.5_s
#
#SBATCH --ntasks=1
##SBATCH --exclusive
#SBATCH --cpus-per-task=8      # Set the desired number of threads here.
#SBATCH --time=1000000:00
#SBATCH --output=out
#SBATCH --

# Use the SLURM_CPUS_PER_TASK variable set automatically by SLURM
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact


source /home/mchan/venv_pyscf/bin/activate
srun --mpi=pmi2 --cpu-bind=cores -o log_%t python3 -u find_the_schedule.py 2>&1
deactivate

