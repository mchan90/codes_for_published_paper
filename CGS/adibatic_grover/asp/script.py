import os
import numpy as np
import shutil
import subprocess
import math

num_tf = 32

n_grover_list = np.arange(1,21,1)

data_dir = './data'
schedule_dir = '../segmented_cgs'

for j in range(len(n_grover_list)):
    n_grover = n_grover_list[j]
    N = 2**n_grover
    t_f_max = N * 10
    dt = 0.5
    folder = data_dir + '/' + str(n_grover)
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print(f"'./{folder} was created.")
    else:
        print(f"./{folder} already exists.")

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
                    st = '#SBATCH --job-name={indx}_a_l'.format(indx=n_grover) +'\n'
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
                elif (line.startswith('n_grover =')):
                    st = 'n_grover = {n_grover}'.format(n_grover=n_grover) +'\n'
                    lines.append(st)
                elif (line.startswith('dt =')):
                    st = 'dt = {dt}'.format(dt=dt) +'\n'
                    lines.append(st)
                elif (line.startswith('num_tf =')):
                    st = 'num_tf = {num_tf}'.format(num_tf=num_tf) +'\n'
                    lines.append(st)
                elif (line.startswith('    ov2 = adiabatic_evolve')):
                    st = '    ov2 = adiabatic_evolve(linear_schedule, t_f, dt, phi, phi_exact)' +'\n'
                    lines.append(st)
                elif (line.startswith("with open('../../on_the_fly/optimal_schedule'")):
                    st = "with open('" + schedule_dir + '/' + str(n_grover) + "/optimal_schedule','r') as file_:" + '\n'
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
                    st = '#SBATCH --job-name={indx}_a_o'.format(indx=n_grover) +'\n'
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
                elif (line.startswith('n_grover =')):
                    st = 'n_grover = {n_grover}'.format(n_grover=n_grover) +'\n'
                    lines.append(st)
                elif (line.startswith('dt =')):
                    st = 'dt = {dt}'.format(dt=dt) +'\n'
                    lines.append(st)
                elif (line.startswith('num_tf =')):
                    st = 'num_tf = {num_tf}'.format(num_tf=num_tf) +'\n'
                    lines.append(st)
                elif (line.startswith('    ov2 = adiabatic_evolve')):
                    st = '    ov2 = adiabatic_evolve(optimal_schedule, t_f, dt, phi, phi_exact)' +'\n'
                    lines.append(st)
                elif (line.startswith("with open('../../on_the_fly/optimal_schedule'")):
                    st = "with open('" + schedule_dir + '/' + str(n_grover) + "/optimal_schedule','r') as file_:" + '\n'
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
    folder_sub = folder_sub + '/exact_optimal'

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
                    st = '#SBATCH --job-name={indx}_a_e'.format(indx=n_grover) +'\n'
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
                elif (line.startswith('n_grover =')):
                    st = 'n_grover = {n_grover}'.format(n_grover=n_grover) +'\n'
                    lines.append(st)
                elif (line.startswith('dt =')):
                    st = 'dt = {dt}'.format(dt=dt) +'\n'
                    lines.append(st)
                elif (line.startswith('num_tf =')):
                    st = 'num_tf = {num_tf}'.format(num_tf=num_tf) +'\n'
                    lines.append(st)
                elif (line.startswith('    ov2 = adiabatic_evolve')):
                    st = '    ov2 = adiabatic_evolve(exact_optimal_schedule, t_f, dt, phi, phi_exact)' +'\n'
                    lines.append(st)
                elif (line.startswith("with open('../../on_the_fly/optimal_schedule'")):
                    st = "with open('" + schedule_dir + '/' + str(n_grover) + "/optimal_schedule','r') as file_:" + '\n'
                    lines.append(st)
                else:
                    lines.append(line)
        fname = folder_sub + '/asp_run.py'
        with open(fname,'w') as file_:
            file_.writelines(lines)
        
        fname = 'asp_run.sh'
        cmd = 'sbatch ' + fname
        subprocess.run(cmd, shell=True, cwd=folder_sub)
