import os
import numpy as np
import shutil
import subprocess
import re

number_pattern = re.compile(r"""
    [+-]?                           # 선택적 부호
    (?:\d+\.\d*|\.\d+|\d+)          # 소수 혹은 정수
    (?:[eE][+-]?\d+)?               # 선택적 지수부
""", re.VERBOSE)


gamma = 0.04
tol_norm = 0.005
dt       = 0.5
nmc      = 10000
m_krylov = 100

indx_list = []

data_dir = '../data'
with open('../find_data_points/out','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        if (int(ls[1])==2):
            continue
        indx_list.append(int(ls[1]))
for j in range(len(indx_list)):
    indx = indx_list[j]
    print(j, indx)
    folder = data_dir + '/' + str(indx_list[j])

    # read the gap
    with open(folder+'/gap_min_finding/log_0','r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# with the gap minimum value')):
                raw_nums = number_pattern.findall(line)
                gap = float(raw_nums[0])
    print(gap)


    folder_sub = folder + '/on_the_fly'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    filename = 'optimal_schedule'

    file_path = os.path.join(folder_sub, filename)
    
    # 있으면 건너뛰기
    if os.path.isfile(file_path):
        print(f"'{file_path}' already exists.")
        continue

    #if not os.path.isfile(file_path):
    #    print(f"'{file_path}' does not exist.")
    #    continue



    with open('./find_the_schedule.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={indx}_s'.format(indx=indx) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/find_the_schedule.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./find_the_schedule.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('gap ')):
                st = 'gap = {gap}'.format(gap=gap) +'\n'
                lines.append(st)
            elif (line.startswith('gamma ')):
                st = 'gamma = {gamma}'.format(gamma=gamma) +'\n'
                lines.append(st)
            elif (line.startswith('tol_norm ')):
                st = 'tol_norm = {tol_norm}'.format(tol_norm=tol_norm) +'\n'
                lines.append(st)
            elif (line.startswith('dt ')):
                st = 'dt = {dt}'.format(dt=dt) +'\n'
                lines.append(st)
            elif (line.startswith('nmc ')):
                st = 'nmc = {nmc}'.format(nmc=nmc) +'\n'
                lines.append(st)
            elif (line.startswith('m_krylov ')):
                st = 'm_krylov = {m_krylov}'.format(m_krylov=m_krylov) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/find_the_schedule.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'find_the_schedule.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
