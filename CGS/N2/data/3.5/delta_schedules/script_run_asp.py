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

bond_length = 3.5
num_tf = 32

with open('../T_asp_w_opt','r') as file_:
    lines = file_.readlines()
    for line in lines:
        if (line.startswith('# gap_est_min')):
            raw_nums = number_pattern.findall(line)
            gap_min = float(raw_nums[0])


gap_min_4 = 1.5238178076515396e-04
t_f_max_4 = 4.0e+05
t_f_max  = t_f_max_4 * gap_min_4 / gap_min

# only for bond_lenght=1.5
if (bond_length<1.6):
    t_f_max = 40

p_min = 1.0
p_max = 2.0

step  = 0.1
n = round((p_max-p_min) /step) + 1 

p_list = [p_min + step * i for i in range(n)]

#p_list = [1.2]

data_dir = './'

for j in range(len(p_list)):
    p = p_list[j]
    print(j, p_list[j])
    folder = data_dir + '{p:.1f}'.format(p=p)

    if (t_f_max<50):
        dt      = 0.01
    else:
        dt = 0.5 # min(t_f_max/50, 1.0)
    folder_sub = folder + '/asp'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    file_path = folder_sub + '/fidelity_'

    if not os.path.isfile(file_path):
        with open('./asp_run.sh','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('#SBATCH --job-name')):
                    st = '#SBATCH --job-name={bond_length:.1f}_{p}'.format(bond_length=bond_length,p=p) +'\n'
                    lines.append(st)
                else:
                    lines.append(line)

        fname = folder_sub + '/asp_run.sh'
        with open(fname,'w') as file_:
            file_.writelines(lines)

        with open('./asp_run.py','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('t_f_max =')):
                    st = 't_f_max = {t_f_max}'.format(t_f_max=t_f_max) +'\n'
                    lines.append(st)
                elif (line.startswith('bond_length =')):
                    st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                    lines.append(st)
                elif (line.startswith('dt =')):
                    st = 'dt = {dt}'.format(dt=dt) +'\n'
                    lines.append(st)
                elif (line.startswith('num_tf =')):
                    st = 'num_tf = {num_tf}'.format(num_tf=num_tf) +'\n'
                    lines.append(st)
                else:
                    lines.append(line)
        fname = folder_sub + '/asp_run.py'
        with open(fname,'w') as file_:
            file_.writelines(lines)
        
        fname = 'asp_run.sh'
        cmd = 'sbatch ' + fname
        subprocess.run(cmd, shell=True, cwd=folder_sub)

# additional opt run
folder = data_dir + 'opt'

if (t_f_max<50):
    dt      = 0.01
else:
    dt = 0.5 # min(t_f_max/50, 1.0)
folder_sub = folder + '/asp'

if not os.path.isdir(folder_sub):
    os.makedirs(folder_sub)
    print(f"'./{folder_sub} was created.")
else:
    print(f"./{folder_sub} already exists.")

file_path = folder_sub + '/fidelity'

if not os.path.isfile(file_path):
    with open('./asp_run.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={bond_length:.1f}_opt'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)

    fname = folder_sub + '/asp_run.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./asp_run.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('t_f_max =')):
                st = 't_f_max = {t_f_max}'.format(t_f_max=t_f_max) +'\n'
                lines.append(st)
            elif (line.startswith('bond_length =')):
                st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            elif (line.startswith('dt =')):
                st = 'dt = {dt}'.format(dt=dt) +'\n'
                lines.append(st)
            elif (line.startswith('num_tf =')):
                st = 'num_tf = {num_tf}'.format(num_tf=num_tf) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/asp_run.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'asp_run.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
