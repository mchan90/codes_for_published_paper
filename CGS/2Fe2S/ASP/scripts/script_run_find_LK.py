import os
import numpy as np
import shutil
import subprocess
import re

number_pattern = re.compile(r"""
    [+-]?                           # optional sign
    (?:\d+\.\d*|\.\d+|\d+)          # decimal or integer
    (?:[eE][+-]?\d+)?               # optional exponent
""", re.VERBOSE)


indx_list = []

data_dir = '../data'
with open('../find_data_points/out','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        if (int(ls[1])==2):
            continue
        indx_list.append(int(ls[1]))


#j_min = 0
#j_max = 20

#j_min = 20
#j_max = 50

j_min = 50
j_max = len(indx_list)
for j in range(j_min,j_max):
    indx = indx_list[j]
    print(j, indx)
    folder = data_dir + '/' + str(indx_list[j])

    # check the existence of the schedule

    folder_sub = folder + '/qz'

    filename = 'optimal_schedule'

    file_path = os.path.join(folder_sub, filename)
    
    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exist.")
        continue

    # read the gap
    folder_sub = folder + '/find_LK'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    filename = 'K_vs_s'

    file_path = os.path.join(folder_sub, filename)
    
    # Skip if it already exists
    if os.path.isfile(file_path):
        print(f"'{file_path}' already exists.")
        continue



    with open('./find_LK.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={indx}_Lk'.format(indx=indx) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/find_LK.sh'
    with open(fname,'w') as file_:
        file_.writelines(lines)

    with open('./find_LK.py','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            lines.append(line)
    fname = folder_sub + '/find_LK.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'find_LK.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
