import os
import numpy as np
import shutil
import subprocess
import math

def next_power_of_10(t_f):
    if t_f <= 0:
        return 1  # adjustable depending on the definition
    # 1) Use logs
    n = math.floor(math.log10(t_f)) + 1
    return 10**n


num_tf = 32
indx_list = []
tf_list = []

data_dir = '../data'
with open('./picked3','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        indx_list.append(int(ls[0]))
        tf_list.append(float(ls[1]))

for j in range(len(indx_list)):
    indx = indx_list[j]
    print(j, indx)
    t_f_max = tf_list[j] * 5
    dt = 1.0 #next_power_of_10(t_f_max)/1000 slower
    #print(t_f_max, dt)
    folder = data_dir + '/' + str(indx_list[j])
    folder_sub = folder + '/asp'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    # linear scheudule
    folder_sub = folder_sub + '/linear'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    file_path = folder_sub + '/fidelity'

    if not os.path.isfile(file_path):
        with open('./asp_run.sh','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('#SBATCH --job-name')):
                    st = '#SBATCH --job-name={indx}_a_l'.format(indx=indx) +'\n'
                    lines.append(st)
                else:
                    lines.append(line)

        fname = folder_sub + '/asp_run.sh'
        with open(fname,'w') as file_:
            file_.writelines(lines)

        with open('./asp_run.py','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('t_f_max =')):
                    st = 't_f_max = {t_f_max}'.format(t_f_max=t_f_max) +'\n'
                    lines.append(st)
                elif (line.startswith('dt =')):
                    st = 'dt = {dt}'.format(dt=dt) +'\n'
                    lines.append(st)
                elif (line.startswith('num_tf =')):
                    st = 'num_tf = {num_tf}'.format(num_tf=num_tf) +'\n'
                    lines.append(st)
                elif (line.startswith('    ov2 = adiabatic_evolve')):
                    st = '    ov2 = adiabatic_evolve(linear_schedule, t_f, dt, phi, fcivec_exact, info_i, info_f)' +'\n'
                    lines.append(st)
                else:
                    lines.append(line)
        fname = folder_sub + '/asp_run.py'
        with open(fname,'w') as file_:
            file_.writelines(lines)
        
        fname = 'asp_run.sh'
        cmd = 'sbatch ' + fname
        subprocess.run(cmd, shell=True, cwd=folder_sub)

    # optimal scheudule
    folder_sub = folder + '/asp'
    folder_sub = folder_sub + '/optimal'

    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    file_path = folder_sub + '/fidelity'

    if not os.path.isfile(file_path):
        with open('./asp_run.sh','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('#SBATCH --job-name')):
                    st = '#SBATCH --job-name={indx}_a_o'.format(indx=indx) +'\n'
                    lines.append(st)
                else:
                    lines.append(line)

        fname = folder_sub + '/asp_run.sh'
        with open(fname,'w') as file_:
            file_.writelines(lines)

        with open('./asp_run.py','r') as file_:
            lines = []
            lines_slurm = file_.readlines()
            for line in lines_slurm:
                if (line.startswith('t_f_max =')):
                    st = 't_f_max = {t_f_max}'.format(t_f_max=t_f_max) +'\n'
                    lines.append(st)
                elif (line.startswith('dt =')):
                    st = 'dt = {dt}'.format(dt=dt) +'\n'
                    lines.append(st)
                elif (line.startswith('num_tf =')):
                    st = 'num_tf = {num_tf}'.format(num_tf=num_tf) +'\n'
                    lines.append(st)
                else:
                    lines.append(line)
        fname = folder_sub + '/asp_run.py'
        with open(fname,'w') as file_:
            file_.writelines(lines)
        
        fname = 'asp_run.sh'
        cmd = 'sbatch ' + fname
        subprocess.run(cmd, shell=True, cwd=folder_sub)
