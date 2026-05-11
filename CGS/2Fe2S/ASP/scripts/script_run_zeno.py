import os
import numpy as np
import shutil
import subprocess


gamma = 0.04
indx_list = []

data_dir = '../data'
with open('../find_data_points/out','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        indx_list.append(int(ls[1]))
#        print(indx_list[-1])

for j in range(len(indx_list)):
    indx = indx_list[j]
    print(j, indx)
    folder = data_dir + '/' + str(indx_list[j])
    folder_sub = folder + '/qz'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    filename = 'DONE'

    file_path = os.path.join(folder_sub, filename)
    
    # Skip if it already exists
    if os.path.isfile(file_path):
        print(f"'{file_path}' already exists.")
        continue



    with open('./zeno_run.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={indx}_z'.format(indx=indx) +'\n'
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
            else:
                lines.append(line)
    fname = folder_sub + '/zeno_run.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'zeno_run.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
