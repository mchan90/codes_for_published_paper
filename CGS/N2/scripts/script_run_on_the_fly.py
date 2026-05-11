import os
import numpy as np
import shutil
import subprocess
import math
import re

number_pattern = re.compile(r"""
    [+-]?                           # optional sign
    (?:\d+\.\d*|\.\d+|\d+)          # decimal or integer
    (?:[eE][+-]?\d+)?               # optional exponent
""", re.VERBOSE)


gamma = 0.04
tol_norm = 0.005
dt       = 0.5
nmc      = 10000
m_krylov = 100

r_min = 0.4
r_max = 3.4

#r_min = 3.6
#r_max = 4.0

step  = 0.1
n = round((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]

#bond_lengths = [0.4]
#bond_lengths = [3.5]

data_dir = '../data'

for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)

    with open(folder+'/T_asp_w_opt','r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# gap_est_min')):
                raw_nums = number_pattern.findall(line)
                gap = float(raw_nums[0])

    folder_sub = folder + '/on_the_fly'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    filename = 'optimal_schedule'

    file_path = os.path.join(folder_sub, filename)
    
    # Skip if it already exists
    if os.path.isfile(file_path):
        print(f"'{file_path}' already exists.")
        #continue



    with open('./find_the_schedule.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={bond_length:.1f}_s'.format(bond_length=bond_length) +'\n'
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
            elif (line.startswith('bond_length =')):
                st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/find_the_schedule.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'find_the_schedule.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
