import os
import numpy as np
import shutil
import subprocess
import math

nmc = 10000
gamma = 0.04
tol_norm = 5e-3
t1_base = 8
dt = 0.5
h_start = 1.0/1024
r = 2.0


n_grover_list = np.arange(1,21,1)

data_dir = './'

for j in range(len(n_grover_list)):
    n_grover = n_grover_list[j]
    folder = data_dir + '/' + str(n_grover)
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print(f"'./{folder} was created.")
    else:
        print(f"./{folder} already exists.")

    file_path = folder + '/optimal_schedule'

    if not os.path.isfile(file_path):
        with open('./grover.sh','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('#SBATCH --job-name')):
                    st = '#SBATCH --job-name={j}_grover'.format(j=n_grover) +'\n'
                    lines.append(st)
                else:
                    lines.append(line)

        fname = folder + '/grover.sh'
        with open(fname,'w') as file_:
            file_.writelines(lines)

        with open('./grover.py','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('n_grover =')):
                    st = 'n_grover = {n_grover}'.format(n_grover=n_grover) +'\n'
                    lines.append(st)
                elif (line.startswith('nmc =')):
                    st = 'nmc = {nmc}'.format(nmc=nmc) +'\n'
                    lines.append(st)
                elif (line.startswith('tol_norm =')):
                    st = 'tol_norm = {tol_norm}'.format(tol_norm=tol_norm) +'\n'
                    lines.append(st)
                elif (line.startswith('gamma =')):
                    st = 'gamma = {gamma}'.format(gamma=gamma) +'\n'
                    lines.append(st)
                elif (line.startswith('dt =')):
                    st = 'dt = {dt}'.format(dt=dt) +'\n'
                    lines.append(st)
                elif (line.startswith('t1_base')):
                    st = 't1_base = {t1_base}'.format(t1_base=t1_base) +'\n'
                    lines.append(st)
                elif (line.startswith('    result = isolate_interval')):
                    st = '    result = isolate_interval(func, s, 1.0, fs, h={h_start}, r={r}, tol=tol_norm)'.format(h_start=h_start,r=r) +'\n'
                    lines.append(st)
                else:
                    lines.append(line)
        fname = folder + '/grover.py'
        with open(fname,'w') as file_:
            file_.writelines(lines)
        
        fname = 'grover.sh'
        cmd = 'sbatch ' + fname
        subprocess.run(cmd, shell=True, cwd=folder)

