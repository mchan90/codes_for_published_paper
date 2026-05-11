import os
import numpy as np
import shutil
import subprocess


ngrids = 101
cores  = 64
n_roots = 50

r_min = 0.4
r_max = 4.0
step  = 0.1
n = int((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]

#bond_lengths = [1.2]

data_dir = '../data'

for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)
    folder_sub = folder + '/est'

    filename = 'T_asp_est'

    file_path = os.path.join(folder_sub, filename)
    
    #if os.path.isfile(file_path):
    #    print(f"'{file_path}' already exists.")
    #    continue

    with open('./grids_run.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={bond_length:.1f}_grid'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            elif (line.startswith('#SBATCH --ntasks')):
                st = '#SBATCH --ntasks={cores}'.format(cores=cores) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/grids_run.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./grids_run.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('bond_length =')):
                st = 'bond_length = {bond_length:.1f}'.format(bond_length=bond_length) +'\n'
                lines.append(st)
            elif (line.startswith('interpol_list')):
                st = 'interpol_list = np.linspace(0,10000,num={ngrids},dtype=int)'.format(ngrids=ngrids) +'\n'
                lines.append(st)
            elif (line.startswith('n_roots')):
                st = 'n_roots = {n_roots}'.format(n_roots=n_roots) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/grids_run.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'grids_run.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
