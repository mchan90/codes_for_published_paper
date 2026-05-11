import os
import numpy as np
import shutil
import subprocess


gamma = 0.05

r_min = 0.4
r_max = 4.0
step  = 0.1
n = int((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]

#bond_lengths = [0.4]

data_dir = '../data'

for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)
    folder_sub = folder + '/qz'

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



    with open('./zeno_run.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={bond_length:.1f}_zeno'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/zeno_run.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./zeno_run.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('gamma')):
                st = 'gamma = {gamma}'.format(gamma=gamma) +'\n'
                lines.append(st)
            elif (line.startswith('bond_length =')):
                st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/zeno_run.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'zeno_run.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
