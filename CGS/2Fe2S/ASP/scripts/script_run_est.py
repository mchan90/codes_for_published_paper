import os
import numpy as np
import shutil
import subprocess


ngrids = 1001
cores  = 64

indx_list = []

data_dir = '../data'

with open('../find_data_points/out','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        indx_list.append(int(ls[1]))
for j in range(len(indx_list)):
    indx = indx_list[j]
    print(j, indx)
    folder = data_dir + '/' + str(indx_list[j])
    folder_sub = folder + '/est'

    filename = 'T_asp_est'

    file_path = os.path.join(folder_sub, filename)
    
    # 파일이 없으면 새로 만들고, 있으면 건너뛰기
    if os.path.isfile(file_path):
        print(f"'{file_path}' already exists.")
        continue



    with open('./grids_run.sh','r') as file_:
        lines = []
        lines_slurm = file_.readlines()
        for line in lines_slurm:
            if (line.startswith('#SBATCH --job-name')):
                st = '#SBATCH --job-name={indx}_grids'.format(indx=indx) +'\n'
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
            if (line.startswith('interpol_list')):
                st = 'interpol_list = np.linspace(0,10000,num={ngrids},dtype=int)'.format(ngrids=ngrids) +'\n'
                lines.append(st)
            else:
                lines.append(line)
    fname = folder_sub + '/grids_run.py'
    with open(fname,'w') as file_:
        file_.writelines(lines)
    
    fname = 'grids_run.sh'
    cmd = 'sbatch ' + fname
    subprocess.run(cmd, shell=True, cwd=folder_sub)
