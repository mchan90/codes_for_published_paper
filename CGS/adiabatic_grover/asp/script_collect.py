import os
import numpy as np
import shutil
import subprocess


n_grover_list = np.arange(1,21,1)
data_dir = './data'


f_crit = 0.75

collected = []
skipped  = [False] * len(n_grover_list)

for j in range(len(n_grover_list)):
    n_grover = n_grover_list[j]
    N = 2**n_grover
    gap = 1/np.sqrt(N)
    folder = data_dir + '/' + str(n_grover)

    file_path = folder + '/asp/linear/fidelity'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue

    tfs = [] 
    fidelities = []
    
    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            ls = line.split()
            tfs.append(float(ls[0]))
            fidelities.append(float(ls[1]))
    print(tfs[0], tfs[-1])
    
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    pchip = PchipInterpolator(tfs, fidelities)

    func = lambda x: pchip(x) - f_crit
    t_asp_linear = brentq(func, tfs[0], tfs[-1])

    file_path = folder + '/asp/optimal/fidelity'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue

    tfs = [] 
    fidelities = []
    
    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            ls = line.split()
            tfs.append(float(ls[0]))
            fidelities.append(float(ls[1]))
    
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    pchip = PchipInterpolator(tfs, fidelities)

    func = lambda x: pchip(x) - f_crit
    t_asp_optimal = brentq(func, tfs[0], tfs[-1])

    
    print(gap, t_asp_linear, t_asp_optimal)


    collected.append([gap, t_asp_linear, t_asp_optimal])

with open('collected_data','w') as file_:
    ind = 0
    for i in range(len(n_grover_list)):
        if (skipped[i]):
            continue
        s = '{:.16e}    {:.16e}    {:.16e}'.format(collected[ind][0],collected[ind][1],collected[ind][2])
        s += '\n'
        file_.write(s)
        ind += 1
    
