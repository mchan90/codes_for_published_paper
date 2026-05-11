#!/bin/bash
#
#SBATCH --job-name=3.5_1.2
#
#SBATCH --ntasks=8
##SBATCH --exclusive
#SBATCH --cpus-per-task=8      # 여기서 원하는 스레드 수를 지정합니다.
#SBATCH --time=1000000:00
#SBATCH --output=out
#SBATCH --

# SLURM이 자동으로 설정하는 SLURM_CPUS_PER_TASK 변수 사용
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact


source /home/mchan/venv_pyscf/bin/activate
srun --mpi=pmi2 --cpu-bind=cores -o log_%t python3 -u asp_run.py 2>&1
deactivate

