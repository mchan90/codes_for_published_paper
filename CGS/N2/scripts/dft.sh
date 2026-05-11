#!/bin/bash
#
#SBATCH --job-name=3.5_dft
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Set the desired number of threads here.
#SBATCH --time=1000000:00
#SBATCH --output=out

# Use the SLURM_CPUS_PER_TASK variable set automatically by SLURM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /home/mchan/venv_pyscf/bin/activate
srun --mpi=pmi2 --cpu-bind=cores python3 -u dft.py
deactivate

