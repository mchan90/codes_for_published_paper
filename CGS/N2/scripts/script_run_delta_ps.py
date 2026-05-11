import os
import numpy as np
import shutil
import subprocess

r_min = 1.5
r_max = 3.9
step  = 0.1
n = int((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]

bond_lengths = [3.9]


data_dir = '../data'
for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)
    folder_sub = folder + '/delta_schedules'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")
#        continue

    with open('./for_deltas/script_run_asp.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('bond_length =')):
                st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/script_run_asp.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./for_deltas/asp_run.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            lines.append(line)
    fname = folder_sub + '/asp_run.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./for_deltas/asp_run.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            lines.append(line)
    fname = folder_sub + '/asp_run.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./for_deltas/make_schedules.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            lines.append(line)
    fname = folder_sub + '/make_schedules.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./for_deltas/make_opt_schedule.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            lines.append(line)
    fname = folder_sub + '/make_opt_schedule.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    fname = 'make_schedules.py'
    cmd = 'python3 ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)

    fname = 'make_opt_schedule.py'
    cmd = 'python3 ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)

    fname = 'script_run_asp.py'
    cmd = 'python3 ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
