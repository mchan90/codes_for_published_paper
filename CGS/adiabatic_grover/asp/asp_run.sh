#!/bin/bash
#
#SBATCH --job-name=14_a_o
#
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1      # Set the desired number of threads here.
#SBATCH --time=1000000:00
#SBATCH --output=out
#SBATCH --

# Use the SLURM_CPUS_PER_TASK variable set automatically by SLURM
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact


source /home/mchan/venv_qiskit/bin/activate
srun --mpi=pmi2 --cpu-bind=cores -o log_%t python3 -u asp_run.py 2>&1
deactivate

