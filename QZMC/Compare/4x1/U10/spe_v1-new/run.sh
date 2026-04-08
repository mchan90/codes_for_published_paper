#!/bin/bash
#
#SBATCH --job-name=spe-v1
#
#SBATCH --ntasks=256
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --time=1000000:00
##SBATCH --mem-per-cpu=100
#SBATCH --output=out
source /home/mchan/venv_qiskit/bin/activate
srun --mpi=pmi2 --cpu-bind=cores python3 -u spe.py
deactivate
