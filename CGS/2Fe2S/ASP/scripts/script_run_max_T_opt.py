import os
import numpy as np
import shutil
import subprocess


eps_x = 1e-6
cores  = 64

indx_list = []

data_dir = '../data'

with open('../find_data_points/out','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        indx_list.append(int(ls[1]))

#indx_list = [0]
#indx_list = [0,2]
for j in range(len(indx_list)):
    indx = indx_list[j]
    print(j, indx)
    folder = data_dir + '/' + str(indx_list[j])
    folder_sub = folder + '/opt_T_max_finding'

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

    file_path = folder + '/qz/optimal_schedule'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exist.")
        continue



    with open('./find_max_T_opt.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={indx}_To'.format(indx=indx) +'\n'
                lines.append(st)
            elif (line.startswith('#SBATCH --ntasks')):
                st = '#SBATCH --ntasks={cores}'.format(cores=cores) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/find_max_T_opt.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./find_max_T_opt.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('eps_x')):
                st = 'eps_x={eps_x}'.format(eps_x=eps_x) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/find_max_T_opt.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'find_max_T_opt.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
