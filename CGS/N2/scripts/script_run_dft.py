import os
import numpy as np
import shutil
import subprocess


r_min = 0.4
r_max = 4.0
step  = 0.1
n = int((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]

fresh_start = True

data_dir = '../data'
for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print(f"'{folder} was created.")
    else:
        print(f"{folder} already exists.")

    folder_sub = folder + '/DFT'
    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'{folder_sub} was created.")
    else:
        print(f"{folder_sub} already exists.")
        if (not fresh_start):
            continue

    with open('./dft.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={bond_length:.1f}_dft'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/dft.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./dft.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('bond_length =')):
                st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/dft.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'dft.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
