#!/bin/bash
#
#SBATCH --job-name=qzmc
#
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --time=1000000:00
#SBATCH --mem-per-cpu=100
#SBATCH --output=out
export OMP_NUM_THREADS=1
source /home/mchan/venv_qiskit/bin/activate
python3 test.py 
deactivate
