#!/bin/bash
#
#SBATCH --job-name=3563_To
#
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1      # 여기서 원하는 스레드 수를 지정합니다.
#SBATCH --time=1000000:00
#SBATCH --output=out

# SLURM이 자동으로 설정하는 SLURM_CPUS_PER_TASK 변수 사용
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact

source /home/mchan/venv_pyscf/bin/activate
srun --mpi=pmi2 --cpu-bind=cores -o log_%t python3 -u find_max_T_opt.py
deactivate

