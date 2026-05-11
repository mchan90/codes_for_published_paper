#!/bin/bash
#
#SBATCH --job-name=3.5_dft
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # 여기서 원하는 스레드 수를 지정합니다.
#SBATCH --time=1000000:00
#SBATCH --output=out

# SLURM이 자동으로 설정하는 SLURM_CPUS_PER_TASK 변수 사용
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /home/mchan/venv_pyscf/bin/activate
srun --mpi=pmi2 --cpu-bind=cores python3 -u dft.py
deactivate

