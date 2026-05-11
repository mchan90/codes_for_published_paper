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


#r_min = 3.6
#r_max = 1.6

r_min = 0.4
r_max = 4.0
step  = 0.1
n = round((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]

#bond_lengths = [0.4]


num_tf = 32

data_dir = '../data'

for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)

    with open(folder+'/T_asp_w_opt','r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# T_asp_est_max')):
                raw_nums = number_pattern.findall(line)
                T_asp_max = float(raw_nums[0])

    t_f_max = T_asp_max * 5
    if (T_asp_max<10):
        dt      = 0.01
    else:
        dt = 0.5 # min(t_f_max/50, 1.0)
    folder_sub = folder + '/asp'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    # linear scheudule
    folder_sub = folder_sub + '/linear'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    #file_path = folder_sub + '/fidelity'
    file_path = folder_sub + '/fidelity__'

    if not os.path.isfile(file_path):
        with open('./asp_run.sh','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('#SBATCH --job-name')):
                    st = '#SBATCH --job-name={bond_length}_a_l'.format(bond_length=bond_length) +'\n'
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
                elif (line.startswith('    schedule = PchipInterpolator(taus, ss)')):
                    st = '    schedule = linear_schedule' +'\n'
                    lines.append(st)
                else:
                    lines.append(line)
        fname = folder_sub + '/asp_run.py'
        with open(fname,'w') as file_:
            file_.writelines(lines)
        
        fname = 'asp_run.sh'
        cmd = 'sbatch ' + fname
        subprocess.run(cmd, shell=True, cwd=folder_sub)

    # optimal scheudule
    folder_sub = folder + '/asp'
    folder_sub = folder_sub + '/optimal'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    #file_path = folder_sub + '/fidelity'
    file_path = folder_sub + '/fidelity__'

    if not os.path.isfile(file_path):
        with open('./asp_run.sh','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('#SBATCH --job-name')):
                    st = '#SBATCH --job-name={bond_length}_a_o'.format(bond_length=bond_length) +'\n'
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
