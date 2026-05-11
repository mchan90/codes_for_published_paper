import os
import numpy as np
import shutil
import subprocess


indx_list = []

data_dir = '../data'
with open('data_points','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        indx_list.append(int(ls[1]))
#        print(indx_list[-1])
for j in range(len(indx_list)):
    indx = indx_list[j]
    print(j, indx)
    folder = data_dir + '/' + str(indx_list[j])
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print(f"'{folder} was created.")
    else:
        print(f"{folder} already exists.")
        continue

    with open('./save_initials.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={indx}_start'.format(indx=indx) +'\n'
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
            if (line.startswith('samp =')):
                st = 'samp = [{indx}]'.format(indx=indx) +'\n'
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
