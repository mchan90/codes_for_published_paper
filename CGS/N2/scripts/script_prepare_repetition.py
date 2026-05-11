import os
import numpy as np
import shutil
import subprocess

r_min = 0.4
r_max = 4.0
step  = 0.1
n = int((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]


data_dir = '../data'
for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)

    with open('./save_initials.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={bond_length:.1f}_start'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder + '/save_initials.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./save_initials.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('bond_length =')):
                st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder + '/save_initials.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    folder_sub = folder + '/est'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    fname = 'save_initials.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder)
